def convert_to_groundtruth_bbox(json_data):
    label_to_action_id = {"Man": 12, "Woman": 74}
    groundtruth_data = []

    video_identifier = json_data["id"]
    for annotation in json_data["annotations"]:
        for result in annotation["result"]:
            action_label = label_to_action_id.get(result["value"]["labels"][0], 0)
            for sequence in result["value"]["sequence"]:
                if sequence["enabled"]:
                    # convert seconds to string with 4 characters
                    time_stamp = "{:.4f}".format(sequence["timestamp"] / 1000000)
                    lt_x, lt_y = sequence["x"] / 100, sequence["y"] / 100
                    rb_x, rb_y = (sequence["x"] + sequence["width"]) / 100, (sequence["y"] + sequence["height"]) / 100
                    entity_id = result.get("id", 0)
                    groundtruth_data.append(f"{video_identifier},{time_stamp},{lt_x:.3f},{lt_y:.3f},{rb_x:.3f},{rb_y:.3f},{action_label},{entity_id}")

    return groundtruth_data


import json

json_data = json.loads(open("annotations.json").read())

lines = convert_to_groundtruth_bbox(json_data)
for line in lines:
    print(line)
