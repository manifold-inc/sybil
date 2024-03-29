import os



class DatasetCatalog:
    def __init__(self):
        # the following dataset utilized for encoding-side alignment learning
        self.audiocap_enc = {
            "target": "dataset.audiocap.AudioCapDataset",
            "params": dict(
                data_path="./data/T-X_pair_data/audiocap/audiocap.json",
                mm_root_path="./data/T-X_pair_data/audiocap/audios",
                embed_path="./data/embed/",
                dataset_type="AudioToText",
            ),
        }

        self.webvid_enc = {
            "target": "dataset.webvid.WebvidDataset",
            "params": dict(
                data_path="./data/T-X_pair_data/webvid/webvid.json",
                mm_root_path="./data/T-X_pair_data/webvid/videos",
                embed_path="./data/embed/",
                dataset_type="VideoToText",
            ),
        }

        self.cc3m_enc = {
            "target": "dataset.cc3m.CC3MDataset",
            "params": dict(
                data_path="./data/T-X_pair_data/cc3m/cc3m.json",
                mm_root_path="./data/T-X_pair_data/cc3m/images",
                embed_path="./data/embed/",
                dataset_type="ImageToText",
            ),
        }

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

        # the following dataset utilized for decoding-side alignment learning.

        self.audiocap_dec = {
            "target": "dataset.audiocap.AudioCapDataset",
            "params": dict(
                data_path="./data/T-X_pair_data/audiocap/audiocap.json",
                mm_root_path="./data/T-X_pair_data/audiocap/audios",
                embed_path="./data/embed/",
                dataset_type="TextToAudio",
            ),
        }

        self.webvid_dec = {
            "target": "dataset.webvid.WebvidDataset",
            "params": dict(
                data_path="./data/T-X_pair_data/webvid/webvid.json",
                mm_root_path="./data/T-X_pair_data/webvid/videos",
                embed_path="./data/embed/",
                dataset_type="TextToVideo",
            ),
        }

        self.cc3m_dec = {
            "target": "dataset.cc3m.CC3MDataset",
            "params": dict(
                data_path="./data/T-X_pair_data/cc3m/cc3m.json",
                mm_root_path="./data/T-X_pair_data/cc3m/images",
                embed_path="./data/embed/",
                dataset_type="TextToImage",
            ),
        }

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

        # the following dataset utilized for instruction tuning, so they are instruction dataset.
        self.audio_instruction = {
            "target": "dataset.T2XT.T2XTInstructionDataset",
            "params": dict(
                data_path="./data/IT_data/T-T+X_data/audio_t2x.json",
                embed_path="./embed/",
                dataset_type="TextToAudio",
            ),
        }

        self.video_instruction = {
            "target": "dataset.T2XT.T2XTInstructionDataset",
            "params": dict(
                data_path="./data/IT_data/T-T+X_data/video_t2x.json",
                embed_path="./embed/",
                dataset_type="TextToVideo",
            ),
        }

        self.image_instruction = {
            "target": "dataset.T2XT.T2XTInstructionDataset",
            "params": dict(
                data_path="./data/IT_data/T-T+X_data/image_t2x.json",
                embed_path="./embed/",
                dataset_type="TextToImage",

            ),
        }

        self.llava_instruction = {
            "target": "dataset.TX2T.TX2TInstructionDataset",
            "params": dict(
                data_path="./data/IT_data/T+X-T_data/llava/llava.json",
                mm_root_path="./data/IT_data/T+X-T_data/llava/images",
                dataset_type="ImageToText",
            ),
        }

        self.alpaca_instruction = {
            "target": "dataset.TX2T.TX2TInstructionDataset",
            "params": dict(
                data_path="./data/IT_data/T+X-T_data/alpaca/alpaca.json",
                dataset_type="TextToText",
            ),
        }

        self.videochat_instruction = {
            "target": "dataset.TX2T.TX2TInstructionDataset",
            "params": dict(
                data_path="./data/IT_data/T+X-T_data/videochat/videochat.json",
                dataset_type="VideoToText",
            ),
        }