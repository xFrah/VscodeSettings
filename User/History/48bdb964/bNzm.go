package database

type StreamItem struct {
	image string
	text  string
}

func (db *appdbimpl) GetStream(userId int64) ([]StreamItem, error) {
	rows, err := db.c.Query("SELECT image, text FROM stream WHERE user_id = ?;", userId)
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