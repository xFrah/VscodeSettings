package database

import "fmt"

func (db *appdbimpl) Follow(followerId, followedId int64) error {
	var exists bool
	err := db.c.QueryRow("SELECT EXISTS(SELECT 1 FROM user WHERE id = ?)", followedId).Scan(&exists)
	if err != nil {
		return err
	}
	if !exists {
		return fmt.Errorf("user does not exist")
	}

	_, err = db.c.Exec("INSERT INTO follows (follower_id, followed_id) VALUES (?, ?)", followerId, followedId)
	if err != nil {
		return err
	}
	return nil
}

func (db *appdbimpl) GetFollowers(userId int64) ([]int64, error) {
	rows, err := db.c.Query(`SELECT follower_id FROM follows WHERE followed_id = ?;`, userId)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var followers []int64
	for rows.Next() {
		var id int64
		err = rows.Scan(&id)
		if err != nil {
			return nil, err
		}
		followers = append(followers, id)
	}
	return followers, nil
}