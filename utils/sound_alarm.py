import winsound

def sound_alarm():
    # Play an alarm sound (Windows-specific)
    frequency = 2500  # Set frequency to 2500 Hz
    duration = 5000  # Set duration to 1000 ms (1 second)
    winsound.Beep(frequency, duration)
