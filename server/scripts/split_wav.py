import numpy as np
import re

from kaldi import util
from kaldi.matrix import SubMatrix
from kaldi.util.table import RandomAccessWaveReader, WaveWriter
from kaldi.feat.wave import WaveData


def extract_segments(wav_rspecifier, segments_rxfilename, wav_wspecifier, min_segment_length=0.1, max_overshoot=0.5):
    print("Extracting segments from wav file...")
    try:
        usage = ''' 
            "Extract segments from a large audio file in WAV format.\n"
            "Usage:  extract-segments [options] <wav-rspecifier> <segments-file> <wav-wspecifier>\n"
            "e.g. extract-segments scp:wav.scp segments ark:- | <some-other-program>\n"
            " segments-file format: each line is either\n"
            "<segment-id> <recording-id> <start-time> <end-time>\n"
            "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5\n"
            "or (less frequently, and not supported in scripts):\n"
            "<segment-id> <wav-file-name> <start-time> <end-time> <channel>\n"
            "where <channel> will normally be 0 (left) or 1 (right)\n"
            "e.g. call-861225-A-0050-0065 call-861225 5.0 6.5 1\n"
            "And <end-time> of -1 means the segment runs till the end of the WAV file\n"
            "See also: extract-feature-segments, wav-copy, wav-to-duration\n"'''

        reader = RandomAccessWaveReader(wav_rspecifier)
        writer = WaveWriter(wav_wspecifier)
        istream = util.io.Input(segments_rxfilename, binary=False)

        num_lines = 0
        num_success = 0

        while 1:
            line = istream.readline()
            if not line:
                break

            num_lines += 1
            split_line = re.split(" |\t|\r", line.strip())

            if len(split_line) != 4 and len(split_line) != 5:
                print("Warning: Invalid line in segments file: " + str(line))
                continue

            segment = split_line[0]
            recording = split_line[1]
            start_str = split_line[2]
            end_str = split_line[3]

            # Parse the start and end times as float values. Segment is ignored if
            # any of end times is malformed.
            start, end = None, None

            start = float(start_str)
            if start is None:
                print(
                    "Warning: Invalid line in segments file [bad start]: " + str(line))
                continue

            end = float(end_str)
            if end is None:
                print(
                    "Warning: Invalid line in segments file [bad end]: " + str(line))
                continue

            # Start time must be non-negative and not greater than the end time,
            # except if the end time is -1.
            if (start < 0 or (end != -1.0 and end <= 0) or ((start >= end) and (end > 0))):
                print(
                    "Warning: Invalid line in segments file [empty or invalid segment]: " + str(line))
                continue

            channel = -1  # -1 means channel is unspecified.
            # If the line has 5 elements, then the 5th element is the channel number.
            if len(split_line) == 5:
                channel = int(split_line[4])
                if not channel or channel < 0:
                    print(
                        "Warning: Invalid line in segments file [bad channel]: " + str(line))
                    continue

            # Check whether the recording ID is in wav.scp; if not, skip the segment.
            if not reader.has_key(recording):
                print("Warning: Could not find recording " +
                      str(recording) + ", skipping segment " + str(segment))
                continue

            wave = reader.value(recording)
            wave_data = wave.data()
            samp_freq = wave.samp_freq  # Sampling fequency.
            num_samp = wave_data.num_cols,  # Number of samples in recording.
            num_chan = wave_data.num_rows  # Number of channels in recording.
            file_length = num_samp[0] / samp_freq  # In seconds.

            # Start must be within the wave data, otherwise skip the segment.
            if start < 0 or start > file_length:
                print("Warning: Segment start is out of file data range [0, " + str(
                    file_length) + "s]; skipping segment '" + str(line) + "'")
                continue

            # End must be less than the file length adjusted for possible overshoot;
            # otherwise skip the segment. end == -1 passes the check.
            if end > file_length + max_overshoot:
                print("Warning: Segment end is too far out of file data range [0," + str(
                    file_length) + "s]; skipping segment '" + str(line) + "'")
                continue

            # Otherwise ensure the end is not beyond the end of data, and default
            # end == -1 to the end of file data.
            if end < 0 or end > file_length:
                end = file_length

            # Skip if segment size is less than the minimum allowed.
            if end - start < min_segment_length:
                print("Warning: Segment " + str(segment) +
                      " too short, skipping it.")
                continue


            # Check that the channel is specified in the segments file for a multi-
            # channel file, and that the channel actually exists in the wave data.
            if channel == -1:
                if num_chan == 1:
                    channel = 0
                else:
                    print("Error: Your data has multiple channels. You must specify the channel in the segments file. "
                          "Skipping segment " + str(segment))
            else:
                if channel >= num_chan:
                    print("Warning: Invalid channel " + str(channel) + " >= " +
                          str(num_chan) + ". Skipping segment " + str(segment))
                    continue

            # Convert endpoints of the segment to sample numbers. Note that the
            # conversion requires a proper rounding.
            start_samp = int(start * samp_freq + 0.5)
            end_samp = int(end * samp_freq + 0.5)

            if end_samp > num_samp[0]:
                end_samp = num_samp[0]

            # Get the range of data from the orignial wave_data matrix.
            segment_matrix = SubMatrix(
                wave_data, channel, 1, start_samp, end_samp - start_samp)
            segment_wave = WaveData.from_data(samp_freq, segment_matrix)
            # Write the range in wave format.
            writer.write(segment, segment_wave)
            num_success += 1

        print("Log: Successfully processed " + str(num_success) +
              " lines out of " + str(num_lines) + " in the segments file. ")
        return 0
    except Exception as e:
        print(str(e))
        return -1
