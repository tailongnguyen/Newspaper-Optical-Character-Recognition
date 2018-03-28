#!/usr/bin/env python
import pika
import time
import selenium
import urllib
import os
import json
import sys
from datetime import datetime
from message_queue import Message
from selenium import webdriver
from selenium.webdriver import ActionChains

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='newests_queue', durable=True)
channel.queue_declare(queue='recognition_queue', durable=True)
print(' [*] Waiting for messages. To exit press CTRL+C')

def get_pages(message):
    url = message['url']
    press = message['name']
    subdomain = message['domain']
    save = os.getcwd() + '/newspaper/' + press
    if subdomain == "vietnam" or subdomain == "lecourrier":
        yield None, None, None
    else:
        if not os.path.isdir(save):
            os.mkdir(save)
        pages = [url + '/files/pages/large/%s.jpg'%str(i) for i in range(1,50)]
        available_pages = []
        for p in pages:
            if urllib.urlopen(p).getcode() == 200:
                print "Getting ", p
                path = save+"/" + message['title'].replace('/', '_') +"_"+ p.split('/')[-1]
                urllib.urlretrieve(p, path)
                yield p, path, str(datetime.now())
            else:
                break

def forward(message, queue_name):
    num = 0
    for page_url, path, date in get_pages(message): 
        if page_url != None:
            new_message = Message(message['domain'], message['url'], message['name'], \
                                  message['title'], page_url, path, date)
            channel.basic_publish(exchange='',
                                routing_key=queue_name,
                                body=json.dumps(new_message.get()),
                                properties=pika.BasicProperties(
                                    delivery_mode = 2, # make message persistent
                                ))
            num += 1
    print(" [x] Sent %d message to %s!" % (num, queue_name))

def callback(channel, method, properties, body):
    print(" [x] Received %r" % body)
    channel.basic_ack(delivery_tag = method.delivery_tag)
    message = json.loads(body)
    forward(message, "recognition_queue")

channel.basic_qos(prefetch_count=1)
channel.basic_consume(callback,
                      queue='newests_queue')

channel.start_consuming()