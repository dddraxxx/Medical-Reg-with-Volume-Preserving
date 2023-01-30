option="${1}" 
case ${option} in 
   1) FILE="${2}" 
      echo "File name is $FILE"
      ;; 
   2) DIR="${2}" 
      echo "Dir name is $DIR"
      ;; 
   ''|*[!0-9]*) 
      python -m debugpy --wait-for-client --listen 5678 $@
      ;; 
   *)
      echo "Listening on port $option"
      python -m debugpy --wait-for-client --listen $@
      ;;
esac 