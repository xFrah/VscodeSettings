from mmaction.datasets import AVADataset

# create ava dataset
ava = AVADataset('data/ava/val_anns.csv', 'data/ava/val_excluded_timestamps.csv', 'data/ava/val.csv', 'data/ava/ava_val_v2.2.csv', 'data/ava/ava_action_list_v2.2.pbtxt', pipeline=[dict(type='SampleAVAFrames', clip_len=32, frame_interval=2, num_clips=1, temporal_jitter=True)], test_mode=True)