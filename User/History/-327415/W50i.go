package api

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"

	"git.sapienzaapps.it/fantasticcoffee/fantastic-coffee-decaffeinated/service/api/reqcontext"
	"github.com/julienschmidt/httprouter"
)

type nameSchema struct {
	Name string `json:"name"`
}

type photoSchema struct {
	Image string `json:"image"`
	Text  string `json:"text"`
}
type UserProfile struct {
	User      string  `json:"user"`
	Followers []int64 `json:"followers"`
	Following []int64 `json:"following"`
}

func (rt *_router) session(w http.ResponseWriter, r *http.Request, ps httprouter.Params, ctx reqcontext.RequestContext) {
	defer r.Body.Close()

	var loginData nameSchema
	err := json.NewDecoder(r.Body).Decode(&loginData)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	userId, err := rt.db.SignIn(loginData.Name)

	w.Header().Set("content-type", "application/json")
	_, _ = w.Write([]byte(fmt.Sprintf(`{"identifier": "%d"}`, userId)))
}

func (rt *_router) testAuth(w http.ResponseWriter, r *http.Request, ps httprouter.Params, ctx reqcontext.RequestContext) {
	_, err := rt.auth(r)
	if err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}

	w.Header().Set("content-type", "application/json")
	_, _ = w.Write([]byte(`{"result": "success"}`))
}

func (rt *_router) nameChange(w http.ResponseWriter, r *http.Request, ps httprouter.Params, ctx reqcontext.RequestContext) {
	defer r.Body.Close()

	userId, err := rt.auth(r)
	if err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}

	var data nameSchema
	err = json.NewDecoder(r.Body).Decode(&data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	err = rt.db.NameChange(userId, data.Name)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("content-type", "application/json")
	_, _ = w.Write([]byte(`{"result": "success"}`))
}

func (rt *_router) auth(r *http.Request) (int64, error) {
	strtoken := r.Header.Get("Authorization")
	token, err := strconv.ParseInt(strtoken, 10, 64)
	if err != nil || strtoken == "" || rt.db.Authenticate(token) != nil {
		return token, errors.New("unauthorized")
	}
	return token, nil
}

func (rt *_router) uploadPhoto(w http.ResponseWriter, r *http.Request, ps httprouter.Params, ctx reqcontext.RequestContext) {
	defer r.Body.Close()

	userId, err := rt.auth(r)
	if err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}

	var data photoSchema
	err = json.NewDecoder(r.Body).Decode(&data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	err = rt.db.UploadPhoto(userId, data.Text, data.Image)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("content-type", "application/json")
	_, _ = w.Write([]byte(`{"result": "success"}`))
}

func (rt *_router) follow(w http.ResponseWriter, r *http.Request, ps httprouter.Params, ctx reqcontext.RequestContext) {
	userId, err := rt.auth(r)
	if err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}

	followedId, err := strconv.ParseInt(ps.ByName("uid"), 10, 64)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	err = rt.db.Follow(userId, followedId) // TODO: check if already following and don't throw error
	if err != nil && err.Error() != "UNIQUE constraint failed: follows.follower_id, follows.followed_id" {
		if err.Error() == "user does not exist" {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("content-type", "application/json")
	_, _ = w.Write([]byte(`{"result": "success"}`))
}

func (rt *_router) unFollow(w http.ResponseWriter, r *http.Request, ps httprouter.Params, ctx reqcontext.RequestContext) {
	userId, err := rt.auth(r)
	if err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}

	followedId, err := strconv.ParseInt(ps.ByName("uid"), 10, 64)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	err = rt.db.UnFollow(userId, followedId)
	// if err != nil {  // TODO: raise everything but the not found
	// 	http.Error(w, err.Error(), http.StatusInternalServerError)
	// 	return
	// }

	w.Header().Set("content-type", "application/json")
	_, _ = w.Write([]byte(`{"result": "success"}`))
}

func (rt *_router) ban(w http.ResponseWriter, r *http.Request, ps httprouter.Params, ctx reqcontext.RequestContext) {
	userId, err := rt.auth(r)
	if err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}

	banId, err := strconv.ParseInt(ps.ByName("uid"), 10, 64)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	err = rt.db.BanUser(userId, banId)
	if err != nil && err.Error() != "UNIQUE constraint failed: blocked.user_id, blocked.blocked_id" {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("content-type", "application/json")
	_, _ = w.Write([]byte(`{"result": "success"}`))
}

func (rt *_router) unBan(w http.ResponseWriter, r *http.Request, ps httprouter.Params, ctx reqcontext.RequestContext) {
	userId, err := rt.auth(r)
	if err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}

	banId, err := strconv.ParseInt(ps.ByName("uid"), 10, 64)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	err = rt.db.UnBanUser(userId, banId)
	// if err != nil {  // TODO: raise everything but the not found
	// 	http.Error(w, err.Error(), http.StatusInternalServerError)
	// 	return
	// }

	w.Header().Set("content-type", "application/json")
	_, _ = w.Write([]byte(`{"result": "success"}`))
}

func (rt *_router) getUserProfile(w http.ResponseWriter, r *http.Request, ps httprouter.Params, ctx reqcontext.RequestContext) {
	userId, err := rt.auth(r)
	if err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}

	profileId, err := strconv.ParseInt(ps.ByName("uid"), 10, 64)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	blocked, err := rt.db.CheckBan(userId, profileId)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if blocked {
		w.WriteHeader(http.StatusForbidden)
		return
	}

	profileName, err := rt.db.GetName(profileId)
	if err == sql.ErrNoRows {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	followers, err1 := rt.db.GetFollowers(profileId)
	following, err2 := rt.db.GetFollowing(profileId)
	stream, err3 := rt.db.GetStream(profileId)

	if err1 != nil {
		http.Error(w, err1.Error(), http.StatusInternalServerError)
		return
	}
	if err2 != nil {
		http.Error(w, err2.Error(), http.StatusInternalServerError)
		return
	}
	if err3 != nil {
		http.Error(w, err3.Error(), http.StatusInternalServerError)
		return
	}

	userProfile := UserProfile{
		User:      profileName,
		Followers: followers,
		Following: following,
		Stream:    stream,
	}

	w.Header().Set("content-type", "application/json")
	_ = json.NewEncoder(w).Encode(userProfile)
}

func (rt *_router) getMyStream(w http.ResponseWriter, r *http.Request, ps httprouter.Params, ctx reqcontext.RequestContext) {
	userId, err := rt.auth(r)
	if err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}

	stream, err := rt.db.GetStream(userId)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("content-type", "application/json")
	_ = json.NewEncoder(w).Encode(stream)
}