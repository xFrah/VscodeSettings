import pyaudio
import wave
import numpy
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.fftpack

notes = [
    ["C0", 16.35, []],
    ["C#0/Db0", 17.32, []],
    ["D0", 18.35, []],
    ["D#0/Eb0", 19.45, []],
    ["E0", 20.60, []],
    ["F0", 21.83, []],
    ["F#0/Gb0", 23.12, []],
    ["G0", 24.50, []],
    ["G#0/Ab0", 25.96, []],
    ["A0", 27.50, []],
    ["A#0/Bb0", 29.14, []],
    ["B0", 30.87, []],
    ["C1", 32.70, []],
    ["C#1/Db1", 34.65, []],
    ["D1", 36.71, []],
    ["D#1/Eb1", 38.89, []],
    ["E1", 41.20, []],
    ["F1", 43.65, []],
    ["F#1/Gb1", 46.25, []],
    ["G1", 49.00, []],
    ["G#1/Ab1", 51.91, []],
    ["A1", 55.00, []],
    ["A#1/Bb1", 58.27, []],
    ["B1", 61.74, []],
    ["C2", 65.41, []],
    ["C#2/Db2", 69.30, []],
    ["D2", 73.42, []],
    ["D#2/Eb2", 77.78, []],
    ["E2", 82.41, []],
    ["F2", 87.31, []],
    ["F#2/Gb2", 92.50, []],
    ["G2", 98.00, []],
    ["G#2/Ab2", 103.83, []],
    ["A2", 110.00, []],
    ["A#2/Bb2", 116.54, []],
    ["B2", 123.47, []],
    ["C3", 130.81, []],
    ["C#3/Db3", 138.59, []],
    ["D3", 146.83, []],
    ["D#3/Eb3", 155.56, []],
    ["E3", 164.81, []],
    ["F3", 174.61, []],
    ["F#3/Gb3", 185.00, []],
    ["G3", 196.00, []],
    ["G#3/Ab3", 207.65, []],
    ["A3", 220.00, []],
    ["A#3/Bb3", 233.08, []],
    ["B3", 246.94, []],
    ["C4", 261.63, []],
    ["C#4/Db4", 277.18, []],
    ["D4", 293.66, []],
    ["D#4/Eb4", 311.13, []],
    ["E4", 329.63, []],
    ["F4", 349.23, []],
    ["F#4/Gb4", 369.99, []],
    ["G4", 392.00, []],
    ["G#4/Ab4", 415.30, []],
    ["A4", 440.00, []],
    ["A#4/Bb4", 466.16, []],
    ["B4", 493.88, []],
    ["C5", 523.25, []],
    ["C#5/Db5", 554.37, []],
    ["D5", 587.33, []],
    ["D#5/Eb5", 622.25, []],
    ["E5", 659.25, []],
    ["F5", 698.46, []],
    ["F#5/Gb5", 739.99, []],
    ["G5", 783.99, []],
    ["G#5/Ab5", 830.61, []],
    ["A5", 880.00, []],
    ["A#5/Bb5", 932.33, []],
    ["B5", 987.77, []],
    ["C6", 1046.50, []],
    ["C#6/Db6", 1108.73, []],
    ["D6", 1174.66, []],
    ["D#6/Eb6", 1244.51, []],
    ["E6", 1318.51, []],
    ["F6", 1396.91, []],
    ["F#6/Gb6", 1479.98, []],
    ["G6", 1567.98, []],
    ["G#6/Ab6", 1661.22, []],
    ["A6", 1760.00, []],
    ["A#6/Bb6", 1864.66, []],
    ["B6", 1975.53, []],
    ["C7", 2093.00, []],
    ["C#7/Db7", 2217.46, []],
    ["D7", 2349.32, []],
    ["D#7/Eb7", 2489.02, []],
    ["E7", 2637.02, []],
    ["F7", 2793.83, []],
    ["F#7/Gb7 ", 2959.96, []],
    ["G7", 3135.96, []],
    ["G#7/Ab7", 3322.44, []],
    ["A7", 3520.00, []],
    ["A#7/Bb7", 3729.31, []],
    ["B7", 3951.07, []],
    ["C8", 4186.01, []],
    ["C#8/Db8", 4434.92, []],
    ["D8", 4698.63, []],
    ["D#8/Eb8", 4978.03, []],
    ["E8", 5274.04, []],
    ["F8", 5587.65, []],
    ["F#8/Gb8", 5919.91, []],
    ["G8", 6271.93, []],
    ["G#8/Ab8", 6644.88, []],
    ["A8", 7040.00, []],
    ["A#8/Bb8", 7458.62, []],
    ["B8", 7902.13, []],
]

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 88200
CHUNK = 5012
RECORD_SECONDS = 0.2
WAVE_OUTPUT_FILENAME = "file.wav"


def normalizeNote(note):
    if len(note) == 2:
        return note[0]
    else:
        return note[0] + note[1]


def getNote(frequency):
    global notes
    for noteIndex in range(0, len(notes)):
        noteData = notes[noteIndex]
        uppperBoundFrequency = noteData[1] * 1.015
        lowerBoundFrequency = noteData[1] * 0.986
        if frequency >= lowerBoundFrequency and frequency <= uppperBoundFrequency:
            return noteData[0]
    return ""


def record():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    for frameIndex in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, "wb")
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b"".join(frames))
    waveFile.close()


def showPlot(timeVector, signal, fftFrequencies, x, y, yRealValues, fft):
    plt.subplot(411)
    plt.plot(timeVector, signal, "g")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.subplot(412)
    plt.plot(fftFrequencies, fft, "r")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Count dbl-sided")
    plt.subplot(413)
    plt.plot(x, y, "b")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Count single-sided")
    plt.subplot(414)
    plt.plot(x, yRealValues, "b")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Count single-sided")
    plt.show()


def detect():
    record()

    fileSampleRate, signal = wavfile.read("file.wav")
    if len(signal.shape) == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    seconds = N / float(fileSampleRate)
    timeSamplesPerSecond = 1.0 / fileSampleRate
    timeVector = scipy.arange(0, seconds, timeSamplesPerSecond)
    fft = abs(scipy.fft.fft(signal))
    fftOneSide = fft[range(N // 2)]
    fftFrequencies = scipy.fftpack.fftfreq(signal.size, timeVector[1] - timeVector[0])
    fftFrequenciesOneSide = fftFrequencies[range(N // 2)]

    realAbsoluteValues = abs(fftOneSide)
    normalizedAbsoluteValues = abs(fftOneSide) / numpy.linalg.norm(abs(fftOneSide))
    x = []
    y = []
    yRealValues = []
    recordedNotes = []
    for frequencyIndex in range(0, len(fftFrequenciesOneSide)):
        if fftFrequenciesOneSide[frequencyIndex] >= 110 and fftFrequenciesOneSide[frequencyIndex] <= 8200:
            x.append(fftFrequenciesOneSide[frequencyIndex])
            y.append(normalizedAbsoluteValues[frequencyIndex])
            yRealValues.append(realAbsoluteValues[frequencyIndex])
            if normalizedAbsoluteValues[frequencyIndex] > 0.25:
                note = getNote(fftFrequenciesOneSide[frequencyIndex])
                if note != "":
                    generalizedNote = normalizeNote(note)
                    if generalizedNote not in recordedNotes:
                        recordedNotes.append(generalizedNote)
    print(recordedNotes)
    showPlot(timeVector, signal, fftFrequencies, x, y, yRealValues, fft)


while True:
    detect()
    #break  # remove to continuously detect it - also remove the showPlot() function then
