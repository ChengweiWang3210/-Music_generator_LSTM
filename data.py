import mido
import numpy as np
import tensorflow as tf
import os
from mido import Message, MidiFile, MidiTrack, MetaMessage
from scipy.stats import bernoulli

# global variables
batch_width = 10  # number of piece chunks in a batch
batch_len = 16 * 8  # length of each chunk: 8 measure with 16 beats per measure
division_len = 16  # 1 measure of division

# Note: We refer to the author's code in this link
# (https://github.com/danieldjohnson/biaxial-rnn-music-composition)
# when trying to transform the input and output data

def toArray(file):
    '''
    Read the midi file and transform to numpy Array with shape=[#beats,#notes=128,2],
    containing the probability of a note being [played,articulated].
    '''
    mid = mido.MidiFile(file)
    timeleft = []
    trackIndex = []
    for track in mid.tracks:
        timeleft.append(track[0].time)
        trackIndex.append(0)
    noteMatrix = []
    time = 0
    note = np.zeros(shape=(128, 2))
    noteMatrix.append(note)

    while True:
        if time % (mid.ticks_per_beat / 4) == (mid.ticks_per_beat / 8):
            preNote = note
            note = [[preNote[x][0], 0] for x in range(128)]
            noteMatrix.append(note)

        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = mid.tracks[i]
                p = trackIndex[i]

                mes = track[p]
                if mes.type == 'time_signature':
                    if mes.numerator not in (2, 4, 8):
                        return False
                elif mes.type == 'note_on':
                    note[mes.note] = [1, 1]
                elif mes.type == 'note_off' or (mes.type == 'note_on' and mes.velocity == 0):
                    note[mes.note] = [0, 0]

                try:
                    timeleft[i] = track[p + 1].time
                    trackIndex[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    noteMatrix = np.asarray(noteMatrix)
    if len(noteMatrix) <= 128+16:
        return False
        
    return noteMatrix


def genBatch(path):
    '''
    TODO:separate data into train and validation folders.
    generate a batch with batch_size=10 piece chunks, each chunk with 8 measures.
    Can be used to generate train/validation set if path specified differently.
    '''
    while True:
    #for _ in range(100):
        batch = []
        names = []
        for fname in os.listdir(path):
            if fname[-4:] not in ('.mid', '.MID'):
                continue
            names.append(fname)

        for i in range(batch_width):
            outArray = toArray(os.path.join(path, np.random.choice(names)))
            while type(outArray) == bool:
                outArray = toArray(os.path.join(path, np.random.choice(names)))

            startPoint = np.random.randint(0, (len(outArray) - batch_len) // division_len) * 16
            batch.append(outArray[startPoint: startPoint + batch_len])

        yield (tf.convert_to_tensor(batch), tf.convert_to_tensor(batch))


def toMidi(noteMatrix):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    tickscale = 100
    lastcmdtick = 0

    prevstate = [[0, 0] for x in range(128)]

    for tick, state in zip(range(noteMatrix.shape[0]), noteMatrix):
        offNotes = []
        onNotes = []
        for i in range(128):
            state[i][0] = bernoulli.rvs(state[i][0])
            state[i][1] = bernoulli.rvs(state[i][1])

            n = state[i]
            p = prevstate[i]

            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)

        for note in offNotes:
            track.append(Message('note_off', time=(tick - lastcmdtick) * tickscale, note=note))
            print("note_off:", Message('note_off', time=(tick - lastcmdtick) * tickscale, note=note))
            lastcmdtick = tick
        for note in onNotes:
            track.append(Message('note_on', time=(tick - lastcmdtick) * tickscale, velocity=40, note=note))
            print("note_on:", Message('note_on', time=(tick - lastcmdtick) * tickscale, velocity=40, note=note))
            lastcmdtick = tick

        prevstate = state

    track.append(MetaMessage('end_of_track', time=0))

    mid.save('output/new_song_1219_5.mid')