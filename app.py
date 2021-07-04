import json
from flask import Flask, flash, render_template, redirect, request, session, make_response, session, redirect
import requests
from urllib.parse import quote
import os
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import time
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = Flask(__name__)

app.secret_key = "lol"

CLIENT_ID = "72305579d392453bad62da72bee5b41d"
CLIENT_SECRET = "84ed762a60d14e7bb8c3f1517447f13e"
PORT = 8080
REDIRECT_URI = "http://127.0.0.1:8080/callback/auth"
SCOPE = 'playlist-modify-private,playlist-modify-public,user-top-read,user-read-recently-played'
API_BASE = 'https://accounts.spotify.com'
SHOW_DIALOG = True

with open('genre.pkl', 'rb') as g:
    genre_data = joblib.load(g)
genre_model = genre_data["model"]

with open('mood.pkl', 'rb') as m:
    mood_data = joblib.load(m)
mood_model = mood_data["model"]


@app.route('/')
def homepage():
    return render_template('homepage.html', invalid="false")


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route("/auth")
def auth():
    auth_url = f'{API_BASE}/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={REDIRECT_URI}&scope={SCOPE}&show_dialog={SHOW_DIALOG}'
    return redirect(auth_url)


@app.route("/callback/auth")
def callback():
    session.clear()
    code = request.args.get('code')
    auth_token_url = f"{API_BASE}/api/token"
    res = requests.post(auth_token_url, data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    })
    res_body = res.json()
    session["toke"] = res_body.get("access_token")
    return redirect("/services")


@app.route("/services")
def services():
    try:
        top_fe, recent_fe, user = get_features()
        top50, recent50 = clean_features(top_fe), clean_features(recent_fe)
        top_genre, recent_genre = get_genre_stats(
            top50), get_genre_stats(recent50)
        top_mood, recent_mood = get_mood_stats(top50), get_mood_stats(recent50)
        display_arr = [user]
        tg, tm, emoji1 = get_chart_data(top_genre, top_mood)
        rg, rm, emoji2 = get_chart_data(recent_genre, recent_mood)
        emojis = emoji1 + emoji2
        return render_template("services.html", user=user, tg=tg, rg=rg, tm=tm, rm=rm, emojis=emojis)
    except:
        return redirect("/auth")


@app.route("/predict", methods=["POST"])
def predict():
    token = SpotifyClientCredentials(
        CLIENT_ID, CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=token)
    inputstr = request.form['getplaylist']
    if inputstr.startswith("https://open.spotify.com/playlist/"):
        id = inputstr[34:56]
        embed = "https://open.spotify.com/embed/playlist/" + id
        playlist_info = sp.playlist(id)
        tracks = playlist_info['tracks']['items']
        while playlist_info['tracks']['next']:
            playlist_info['tracks'] = sp.next(playlist_info['tracks'])
            tracks.extend(playlist_info['tracks']['items'])
        trackid = []
        for i in range(len(tracks)):
            trackid.append(tracks[i]['track']['id'])
        features = []
        for i in range(0, len(trackid), 100):
            features += sp.audio_features(trackid[i:i+100])
        clean = clean_features(features)
        genre_data = get_genre_stats(clean)
        mood_data = get_mood_stats(clean)
        tg, tm, emojis = get_chart_data(genre_data, mood_data)
        print(tg, tm, emojis)
        return render_template("playlist.html", playlist=embed, tg=tg, tm=tm, emojis=emojis)
    else:
        return render_template('homepage.html', invalid="true")


def get_chart_data(genre, mood):
    genre_labels = [i[0] for i in genre]
    genre_items = [i[1] for i in genre]
    tg = [genre_items, genre_labels]
    mood_labels = [i[0] for i in mood]
    mood_items = [i[1] for i in mood]
    tm = [mood_items, mood_labels]
    largest = [tg[1][tg[0].index(max(tg[0]))], tm[1][tm[0].index(max(tm[0]))]]
    emojis = [getemojis(l) for l in largest]
    return tg, tm, emojis


def getemojis(largest):
    if largest == "rock":
        return 129311
    elif largest == "jazz":
        return 127927
    elif largest == "classical":
        return 127931
    elif largest == "pop":
        return 127928
    elif largest == "hiphop":
        return 129336
    elif largest == "sad":
        return 128532
    elif largest == "happy":
        return 128522
    elif largest == "angry":
        return 128545
    elif largest == "calm":
        return 129496


def get_features():
    sp = spotipy.Spotify(auth=session['toke'])
    user = sp.current_user()
    top = sp.current_user_top_tracks(limit=50)['items']
    recent = sp.current_user_recently_played(limit=50)['items']
    topid = [t['id'] for t in top]
    recentid = [r['track']['id'] for r in recent]
    top_fe = sp.audio_features(topid)
    recent_fe = sp.audio_features(recentid)
    return top_fe, recent_fe, user['display_name']


def clean_features(full_features):
    features = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    clean_features = []
    for i in range(len(full_features)):
        clean_features.append([full_features[i][j] for j in features])
    return clean_features


def get_genre_stats(features):
    genre = list(genre_model.predict(features).tolist())
    genre_list = [[t, genre.count(t)] for t in list(set(genre))]
    return genre_list


def get_mood_stats(features):
    mood = list(mood_model.predict(features).tolist())
    mood_list = [[t, mood.count(t)] for t in list(set(mood))]
    return mood_list


def get_token(session):
    token_valid = False
    token_info = session.get("token_info", {})
    if not (session.get('token_info', False)):
        token_valid = False
        return token_info, token_valid
    now = int(time.time())
    is_token_expired = session.get('token_info').get('expires_at') - now < 60
    if (is_token_expired):
        sp_oauth = spotipy.oauth2.SpotifyOAuth(
            client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)
        token_info = sp_oauth.refresh_access_token(
            session.get('token_info').get('refresh_token'))
    token_valid = True
    return token_info, token_valid


if __name__ == "__main__":
    app.run(debug=True, port=PORT)