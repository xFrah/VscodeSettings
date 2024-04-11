package database

func (db *appdbimpl) BanUser(userId, banId int64) error {
	_, err := db.c.Exec(`INSERT INTO blocked (user_id, blocked_id) VALUES (?, ?);`, userId, banId)
	if err != nil {
		return err
	}
	return nil
}

func (db *appdbimpl) UnBanUser(userId, banId int64) error {
	_, err := db.c.Exec(`DELETE FROM blocked WHERE user_id = ? AND blocked_id = ?;`, userId, banId)
	if err != nil {
		return err
	}
	return nil
}

func (db *appdbimpl) CheckBan(userId, profileId int64) (bool, error) {
	var blocked bool
	err := db.c.QueryRow(`SELECT EXISTS(SELECT 1 FROM blocked WHERE user_id = ? AND blocked_id = ?);`, userId, profileId).Scan(&blocked)
	if err != nil {
		return false, err
	}
	return blocked, nil
}
