package database

// struct StreamItem
type StreamItem struct {
	imaage string
	text   string
}

func (db *DB) GetStream(userId int64) ([]StreamItem, error) {
	rows, err := db.Query("SELECT image, text FROM stream WHERE user_id = ?;", userId)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var stream []StreamItem
	for rows.Next() {
		var item StreamItem
		err := rows.Scan(&item.image, &item.text)
		if err != nil {
			return nil, err
		}
		stream = append(stream, item)
	}
	return stream, nil
}