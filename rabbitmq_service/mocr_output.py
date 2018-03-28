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

channel.queue_declare(queue='mocr_output', durable=True)

print(' [*] Waiting for messages. To exit press CTRL+C')

def callback(channel, method, properties, body):
    print(" [x] Received %r" % json.loads(body)['path'])
    channel.basic_ack(delivery_tag = method.delivery_tag)
    message = json.loads(body)
    with codecs.open('extracted_text/'+'_'.join(message['path'].split('/')[-2:]) + '.txt', 'w', 'utf-8') as textfile:
        textfile.write(message['text'])
        textfile.close()

channel.basic_qos(prefetch_count=1)
channel.basic_consume(callback,
                      queue='mocr_output')

channel.start_consuming()