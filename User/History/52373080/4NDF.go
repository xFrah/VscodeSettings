/*
Package database is the middleware between the app database and the code. All data (de)serialization (save/load) from a
persistent database are handled here. Database specific logic should never escape this package.

To use this package you need to apply migrations to the database if needed/wanted, connect to it (using the database
data source name from config), and then initialize an instance of AppDatabase from the DB connection.

For example, this code adds a parameter in `webapi` executable for the database data source name (add it to the
main.WebAPIConfiguration structure):

	DB struct {
		Filename string `conf:""`
	}

This is an example on how to migrate the DB and connect to it:

	// Start Database
	logger.Println("initializing database support")
	db, err := sql.Open("sqlite3", "./foo.db")
	if err != nil {
		logger.WithError(err).Error("error opening SQLite DB")
		return fmt.Errorf("opening SQLite: %w", err)
	}
	defer func() {
		logger.Debug("database stopping")
		_ = db.Close()
	}()

Then you can initialize the AppDatabase and pass it to the api package.
*/
package database

import (
	"database/sql"
	"errors"
	"fmt"
)

// AppDatabase is the high level interface for the DB
type AppDatabase interface {
	GetName(userId int64) (string, error)
	Authenticate(userId int64) error
	SignIn(user string) (int64, error)
	NameChange(userId int64, newName string) error
	UploadPhoto(userId int64, text string, link string) error
	Follow(followerId, followedId int64) error
	UnFollow(followerId, followedId int64) error
	BanUser(userId, banId int64) error
	UnBanUser(userId, banId int64) error
	CheckBan(userId, blockedId int64) (bool, error)
	GetFollowers(userId int64) ([]int64, error)
	GetFollowing(userId int64) ([]int64, int)
	Ping() error
}

type appdbimpl struct {
	c *sql.DB
}

// New returns a new instance of AppDatabase based on the SQLite connection `db`.
// `db` is required - an error will be returned if `db` is `nil`.
func New(db *sql.DB) (AppDatabase, error) {
	if db == nil {
		return nil, errors.New("database is required when building a AppDatabase")
	}

	// Check if table exists. If not, the database is empty, and we need to create the structure
	var tableName string
	var err1, err2, err3, err4, err5, err6, err7 error
	err1 = db.QueryRow(`SELECT name FROM sqlite_master WHERE type='table' AND name='user';`).Scan(&tableName)
	if errors.Is(err1, sql.ErrNoRows) {
		sqlStmt := `CREATE TABLE user (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE);`
		_, err2 = db.Exec(sqlStmt)
		sqlStmt = `CREATE TABLE post (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, file TEXT NOT NULL, message TEXT NOT NULL, likes INTEGER NOT NULL DEFAULT 0, FOREIGN KEY (user_id) REFERENCES user(id));`
		_, err3 = db.Exec(sqlStmt)
		sqlStmt = `CREATE TABLE comment (id INTEGER PRIMARY KEY AUTOINCREMENT, post_id INTEGER NOT NULL, message TEXT NOT NULL, likes INTEGER NOT NULL DEFAULT 0, FOREIGN KEY (post_id) REFERENCES post(id));`
		_, err4 = db.Exec(sqlStmt)
		sqlStmt = `CREATE TABLE like (user_id INTEGER NOT NULL, post_id INTEGER NOT NULL, FOREIGN KEY (user_id) REFERENCES user(id), FOREIGN KEY (post_id) REFERENCES post(id), PRIMARY KEY (user_id, post_id));`
		_, err5 = db.Exec(sqlStmt)
		sqlStmt = `CREATE TABLE follows (follower_id INTEGER NOT NULL, followed_id INTEGER NOT NULL, FOREIGN KEY (follower_id) REFERENCES user(id), FOREIGN KEY (followed_id) REFERENCES user(id), PRIMARY KEY (follower_id, followed_id));`
		_, err6 = db.Exec(sqlStmt)
		sqlStmt = `CREATE TABLE blocked (user_id INTEGER NOT NULL, blocked_id INTEGER NOT NULL, FOREIGN KEY (user_id) REFERENCES user(id), FOREIGN KEY (blocked_id) REFERENCES user(id), PRIMARY KEY (user_id, blocked_id));`
		_, err7 = db.Exec(sqlStmt)
		if err2 != nil || err3 != nil || err4 != nil || err5 != nil || err6 != nil || err7 != nil {
			return nil, fmt.Errorf("error creating database structure.")
		}
	}

	return &appdbimpl{
		c: db,
	}, nil
}

func (db *appdbimpl) Ping() error {
	return db.c.Ping()
}
