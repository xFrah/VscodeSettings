package database

func (db *DB) GetStream(userId int64) ([]*Stream, error) {
	