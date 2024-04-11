package database

// struct StreamItem
type StreamItem struct {
	UserId int64 `json:"user_id"`

func (db *DB) GetStream(userId int64) ([]*Stream, error) {
