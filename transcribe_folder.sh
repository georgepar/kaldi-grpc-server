#!/usr/bin/env bash
cmd(){ echo `basename $0`; }
usage(){
    echo "\
    `cmd` [OPTION...]
    -i, --input; Input wav folder
    -o, --output; Output transcription folder
    -p, --port; ASR port [default: 50051]
    -h, --host; ASR host [default: localhost]
    -j, --njobs; Number of cores for parallel execution [default: `grep -c ^processor /proc/cpuinfo`]
    " | column -t -s ";"
}

print_usage(){
    usage;
    exit 2;
}

abnormal_exit(){
    usage;
    exit 1;
}

SHORT_OPTS=i:o:p:h:j:
LONG_OPTS=input:,output:,port:,host:,njobs:

OPTIONS=`getopt -o ${SHORT_OPTS} --long ${LONG_OPTS} -n "transcribe_folder.sh" -- "$@"`

if [ $? != 0 ] ; then abnormal_exit; fi


INPUT_FOLDER=
OUTPUT_FOLDER=
PORT=50051
HOST=localhost
NJOBS=`grep -c ^processor /proc/cpuinfo`

while true; do
  case "$1" in
    -i | --input ) INPUT_FOLDER="$2"; shift 2 ;;
    -o | --output ) OUTPUT_FOLDER="$2"; shift 2 ;;
    -p | --port ) PORT="$2"; shift 2 ;;
    -h | --host ) HOST="$2"; shift 2 ;;
    -j | --njobs ) NJOBS="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

echo $INPUT_FOLDER
echo $OUTPUT_FOLDER
echo $PORT
echo $HOST
echo $NJOBS



if [ -z "$INPUT_FOLDER" ]
then
    abnormal_exit
fi

if [ -z "$OUTPUT_FOLDER" ]
then
    abnormal_exit
fi


transcribe_wav() {
    wav=$1
    host=$2
    port=$3
    outf=$4
    base_wav=${wav%.*}
    base_wav=${base_wav##*/}
    kaldigrpc-transcribe --host $host --port $port --streaming $wav > ${outf}/${base_wav}.txt
}

export -f transcribe_wav

parallel -j${NJOBS}  transcribe_wav {} $HOST $PORT $OUTPUT_FOLDER ::: $(find ${INPUT_FOLDER} -name "*.wav")
