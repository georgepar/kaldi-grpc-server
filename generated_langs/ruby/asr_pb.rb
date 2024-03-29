# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: asr.proto

require 'google/protobuf'

Google::Protobuf::DescriptorPool.generated_pool.build do
  add_file("asr.proto", :syntax => :proto3) do
    add_message "ilsp.spmd.asr.RecognizeRequest" do
      optional :config, :message, 1, "ilsp.spmd.asr.RecognitionConfig"
      optional :audio, :bytes, 2
    end
    add_message "ilsp.spmd.asr.StreamingRecognizeRequest" do
      oneof :streaming_request do
        optional :streaming_config, :message, 1, "ilsp.spmd.asr.StreamingRecognitionConfig"
        optional :audio_content, :bytes, 2
      end
    end
    add_message "ilsp.spmd.asr.RecognitionConfig" do
      optional :encoding, :enum, 1, "ilsp.spmd.asr.AudioEncoding"
      optional :sample_rate_hertz, :int32, 2
      optional :language_code, :string, 3
      optional :max_alternatives, :int32, 4
      optional :audio_channel_count, :int32, 7
      optional :enable_word_time_offsets, :bool, 8
      optional :enable_automatic_punctuation, :bool, 11
      optional :enable_separate_recognition_per_channel, :bool, 12
      optional :model, :string, 13
    end
    add_message "ilsp.spmd.asr.StreamingRecognitionConfig" do
      optional :config, :message, 1, "ilsp.spmd.asr.RecognitionConfig"
      optional :interim_results, :bool, 2
    end
    add_message "ilsp.spmd.asr.RecognizeResponse" do
      repeated :results, :message, 1, "ilsp.spmd.asr.SpeechRecognitionResult"
    end
    add_message "ilsp.spmd.asr.SpeechRecognitionResult" do
      repeated :alternatives, :message, 1, "ilsp.spmd.asr.SpeechRecognitionAlternative"
      optional :channel_tag, :int32, 2
      optional :audio_processed, :float, 3
    end
    add_message "ilsp.spmd.asr.SpeechRecognitionAlternative" do
      optional :transcript, :string, 1
      optional :confidence, :float, 2
      repeated :words, :message, 3, "ilsp.spmd.asr.WordInfo"
    end
    add_message "ilsp.spmd.asr.WordInfo" do
      optional :start_time, :float, 1
      optional :end_time, :float, 2
      optional :word, :string, 3
    end
    add_message "ilsp.spmd.asr.StreamingRecognizeResponse" do
      repeated :results, :message, 1, "ilsp.spmd.asr.StreamingRecognitionResult"
    end
    add_message "ilsp.spmd.asr.StreamingRecognitionResult" do
      repeated :alternatives, :message, 1, "ilsp.spmd.asr.SpeechRecognitionAlternative"
      optional :is_final, :bool, 2
      optional :channel_tag, :int32, 5
      optional :audio_processed, :float, 6
    end
    add_enum "ilsp.spmd.asr.AudioEncoding" do
      value :ENCODING_UNSPECIFIED, 0
      value :LINEAR_PCM, 1
      value :FLAC, 2
      value :MULAW, 3
      value :ALAW, 20
    end
  end
end

module Ilsp
  module Spmd
    module Asr
      RecognizeRequest = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("ilsp.spmd.asr.RecognizeRequest").msgclass
      StreamingRecognizeRequest = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("ilsp.spmd.asr.StreamingRecognizeRequest").msgclass
      RecognitionConfig = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("ilsp.spmd.asr.RecognitionConfig").msgclass
      StreamingRecognitionConfig = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("ilsp.spmd.asr.StreamingRecognitionConfig").msgclass
      RecognizeResponse = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("ilsp.spmd.asr.RecognizeResponse").msgclass
      SpeechRecognitionResult = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("ilsp.spmd.asr.SpeechRecognitionResult").msgclass
      SpeechRecognitionAlternative = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("ilsp.spmd.asr.SpeechRecognitionAlternative").msgclass
      WordInfo = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("ilsp.spmd.asr.WordInfo").msgclass
      StreamingRecognizeResponse = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("ilsp.spmd.asr.StreamingRecognizeResponse").msgclass
      StreamingRecognitionResult = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("ilsp.spmd.asr.StreamingRecognitionResult").msgclass
      AudioEncoding = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("ilsp.spmd.asr.AudioEncoding").enummodule
    end
  end
end
