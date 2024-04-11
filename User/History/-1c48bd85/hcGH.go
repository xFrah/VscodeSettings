package service

type nameSchema struct {
	Name string `json:"name"`
}

type photoSchema struct {
	Image string `json:"image"`
	Text  string `json:"text"`
}

type StreamItem struct {
	image string
	text  string
}

type UserProfile struct {
	User      string  `json:"user"`
	Followers []int64 `json:"followers"`
	Following []int64 `json:"following"`
	Stream    []StreamItem `json:"stream"`
}