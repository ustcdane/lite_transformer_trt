kill -9 $(ps -ef|grep -E 'python3' | grep -v grep|awk '{print $2}')
