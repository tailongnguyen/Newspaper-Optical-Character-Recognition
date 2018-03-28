#!/usr/bin/env python
import pika
import time
import selenium
import urllib
import os
import json
import sys
import codecs
import requests
from datetime import datetime
from message_queue import Message
from selenium import webdriver
from selenium.webdriver import ActionChains

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='recognition_queue', durable=True)
channel.queue_declare(queue='mocr_output', durable=True)

threshold = {}
t_10 = ['thethaovanhoa', 'baoanhdantocmiennui', 'baodaknong', 'baokiengiang', 'baolamdong', 'baoninhthuan',\
        'baotintuc', 'baotuyenquang', 'baovinhphuc', 'baoyenbai', 'bnews', 'khoahoctoday', 'phapluattp']
t_8 = ['baobacgiang', 'baobacninh', 'baocantho', 'baodongnai']
t_5 = ['baodienbien', 'baohagiang', 'baohaiphong', 'baohatinh', 'baolaichau', 'baothaibinh', 'baothanhhoa']        
for t in t_10:
    threshold[t] = '10'
for t in t_8:
    threshold[t] = '8'
for t in t_5:
    threshold[t] = '5'

print(' [*] Waiting for messages. To exit press CTRL+C')

def forward(message, queue_name):
    path = message['path']
    subdomain = message['domain']
    r = requests.post('http://localhost:1234', files={'file': open(path, 'rb'), \
                                                      'threshold': threshold[subdomain]})
    if r.status_code == 200:
        text = json.loads(r.text)['text']
        new_message = Message(message['domain'], message['url'], message['name'], \
                                    message['title'], message['link_image'], path,\
                                    message['Datetime_crawl'], str(datetime.now()), text)
        channel.basic_publish(exchange='',
                            routing_key=queue_name,
                            body=json.dumps(new_message.get()),
                            properties=pika.BasicProperties(
                                delivery_mode = 2, # make message persistent
                            ))

def callback(channel, method, properties, body):
    print(" [x] Received %r" % body)
    channel.basic_ack(delivery_tag = method.delivery_tag)
    message = json.loads(body)
    forward(message, "mocr_output")

channel.basic_qos(prefetch_count=1)
channel.basic_consume(callback,
                      queue='recognition_queue')

channel.start_consuming()