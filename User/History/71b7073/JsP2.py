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
                    entity_id = sequence.get('id', 0)
                    groundtruth_data.append(f"{video_identifier},{time_stamp},{lt_x:.3f},{lt_y:.3f},{rb_x:.3f},{rb_y:.3f},{action_label},{entity_id}")

    return groundtruth_data