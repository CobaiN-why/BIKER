#! /bin/bash

nohup uwsgi --ini /home/ubuntu/BIKER/BIKER_pages/main/uwsgi.ini > output 2>&1 &

echo "biker search start..."

exit 0
