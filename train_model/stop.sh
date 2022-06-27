#kill -9 $(ps -ef|grep -E 'pythondaniel' | grep -v grep|awk '{print $2}')
kill -9 $(ps -ef|grep -E 'python' | grep -v grep|awk '{print $2}')
