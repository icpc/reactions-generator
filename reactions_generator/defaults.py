class Defaults:
    title = "UNI"
    subtitle = "Subtitle"
    hashtag = "#hashtag"
    task = "A"
    time = 1000000
    outcome = "AC"
    success = True
    rank_before = 100
    rank_after = 1
    logo_source = "example/logo.png"
    webcam_source = "example/reaction.mp4"
    screen_source = "example/screen.mp4"
    background_source = "example/background_vertical.png"
    background_source_h = "example/background-moscow.png"
#   default vertical story
    background_position = (0,0,1080,1920)   
    card_position = (40,807,1000,306)
    webcam_position = (40,194.5,1000,562.5)
    screen_position = (40,1163,1000,562.5)
#   default horizontal story with webcam in bottom left and card in bottom right
#    background_position_h = (0,0,1920,1080)
#    card_position_h = (904,758,1000,306)
#    webcam_position_h = (0,0,1920,1080)
#    screen_position_h = (16,704,640,360)
#   default horizontal story with webcam in top right left and card in top left
    background_position_h = (0,0,1920,1080)
    card_position_h = (16,16,1000,306)
    webcam_position_h = (0,0,1920,1080)
    screen_position_h = (1264,16,640,360)

    success_audio_path = "example/success.mp3"
    fail_audio_path = "example/fail.mp3"
    fps = 30
    duration_seconds = 60
    vcodec = "libx264"
    acodec = "aac"
    output_directory = "out"
    sound = True
    output_path = f"{output_directory}/output.mp4"
