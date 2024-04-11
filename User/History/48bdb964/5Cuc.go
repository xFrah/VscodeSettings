package database

// struct StreamItem
type StreamItem struct {
	imaage string
	text   string
}

func (db *DB) GetStream(userId int64) ([]*Stream, error) {
