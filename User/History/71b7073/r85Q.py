def format_timestamp(time_in_seconds):
    """Assuming time in seconds and converting it to a string with four characters."""
    # This simple conversion assumes time_in_seconds is always below 1 minute for simplicity
    total_seconds = int(time_in_seconds * 100)  # Convert to centiseconds for uniqueness
    return f"{total_seconds:04d}"

def convert_to_groundtruth_bbox(json_data):
    label_to_action_id = {"Man": 12, "Woman": 74}
    groundtruth_data = []

    video_identifier = json_data["id"]
    for annotation in json_data['annotations']:
        for result in annotation['result']:
            action_label = label_to_action_id.get(result['value']['labels'][0], 0)
            for sequence in result['value']['sequence']:
                if sequence['enabled']:
                    time_stamp = format_timestamp(sequence['time'])
                    lt_x, lt_y = sequence['x'] / 100, sequence['y'] / 100
                    rb_x, rb_y = (sequence['x'] + sequence['width']) / 100, (sequence['y'] + sequence['height']) / 100
                    entity_id = result.get('id', 0)
                    groundtruth_data.append(f"{video_identifier},{time_stamp},{lt_x:.3f},{lt_y:.3f},{rb_x:.3f},{rb_y:.3f},{action_label},{entity_id}")

    return groundtruth_data

json_data = {
  "id": 1,
  "data": {
    "video": "/data/upload/1/1380ce26-dash2.mp4"
  },
  "annotations": [
    {
      "id": 1,
      "created_username": " fdimonaco309@hotmail.com, 1",
      "created_ago": "7 minutes",
      "completed_by": {
        "id": 1,
        "first_name": "",
        "last_name": "",
        "avatar": None,
        "email": "fdimonaco309@hotmail.com",
        "initials": "fd"
      },
      "result": [
        {
          "value": {
            "framesCount": 1270,
            "duration": 50.766667,
            "sequence": [
              {
                "frame": 1,
                "enabled": True,
                "rotation": 0,
                "x": 27.609890109890113,
                "y": 28.998778998779002,
                "width": 21.016483516483518,
                "height": 37.11843711843713,
                "time": 0.04
              }
            ],
            "labels": [
              "Man"
            ]
          },
          "id": "l7ajDnPC5U",
          "from_name": "box",
          "to_name": "video",
          "type": "videorectangle",
          "origin": "manual"
        },
        {
          "value": {
            "framesCount": 1270,
            "duration": 50.766667,
            "sequence": [
              {
                "frame": 1,
                "enabled": True,
                "rotation": 0,
                "x": 58.104395604395606,
                "y": 19.963369963369964,
                "width": 29.39560439560439,
                "height": 54.21245421245422,
                "time": 0.04
              }
            ],
            "labels": [
              "Woman"
            ]
          },
          "id": "HSNACKbVbr",
          "from_name": "box",
          "to_name": "video",
          "type": "videorectangle",
          "origin": "manual"
        },
        {
          "value": {
            "framesCount": 1270,
            "duration": 50.766667,
            "sequence": [
              {
                "frame": 1,
                "enabled": True,
                "rotation": 0,
                "x": 6.456043956043957,
                "y": 10.683760683760683,
                "width": 17.71978021978022,
                "height": 38.339438339438345,
                "time": 0.04
              },
              {
                "x": 5.631868131868134,
                "y": 21.67277167277168,
                "width": 17.71978021978022,
                "height": 38.339438339438345,
                "rotation": 0,
                "frame": 2,
                "enabled": True,
                "time": 0.08
              }
            ],
            "labels": [
              "Man"
            ]
          },
          "id": "TsFBkxBhsK",
          "from_name": "box",
          "to_name": "video",
          "type": "videorectangle",
          "origin": "manual"
        }
      ],
      "was_cancelled": False,
      "ground_truth": False,
      "created_at": "2024-02-22T19:12:56.708551Z",
      "updated_at": "2024-02-22T19:20:04.654546Z",
      "draft_created_at": "2024-02-22T19:12:34.917009Z",
      "lead_time": 165.54200000000003,
      "import_id": None,
      "last_action": None,
      "task": 1,
      "project": 1,
      "updated_by": 1,
      "parent_prediction": None,
      "parent_annotation": None,
      "last_created_by": None
    }
  ],
  "predictions": []
}

lines = convert_to_groundtruth_bbox(json_data)
for line in lines:
    print(line)