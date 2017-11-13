#!/usr/bin/env python
import pika
import sys
import selenium
import time
import json
import redis
from selenium import webdriver
from message_queue import Message


connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
redisServer = redis.StrictRedis(host='localhost', port=6379, db=0)

channel.queue_declare(queue='newests_queue', durable=True)

def add_to_queue(message):
    channel.basic_publish(exchange='',
                        routing_key='newests_queue',
                        body=message,
                        properties=pika.BasicProperties(
                            delivery_mode = 2, # make message persistent
                        ))
    print(" [x] Sent %r" % message)

def check(p, newest_link):
    if redisServer.get(p) is None:
        redisServer.set(p, newest_link)
        return True
    else:
        if redisServer.get(p) == newest_link:
            return False
        else:
            return True
        
def get_newest(domain):
    # driver = webdriver.PhantomJS("/home/tailongnguyen/Downloads/phantomjs-2.1.1-linux-x86_64/bin/phantomjs",
                    # service_args=['--load-images=false'])
    driver = webdriver.PhantomJS("/home/long/aeh16/phantomjs-2.1.1-linux-x86_64/bin/phantomjs",
                    service_args=['--load-images=false'])
    driver.get(domain)
    press_tag = driver.find_elements_by_css_selector('li.anphamTTXVN a')
    urls = [a.get_attribute("href") for a in press_tag]
    press = [a.text for a in press_tag]
    for (p, url) in zip(press, urls):
        driver.get(url)
        # print url
        length = 0
        while length == 0:
            try:
                sobao = driver.find_elements_by_css_selector('div#listContent li')
                length = len(sobao)
            except selenium.common.exceptions.NoSuchElementException:
                print "No elements found!"
        newest = sobao[-1]
        newest_link = newest.find_element_by_css_selector('a').get_attribute('href')
        newest_name  = newest.find_element_by_css_selector('div.blockTitle').text
        subdomain = url.split('/')[2].split('.')[0]
        if check(p, newest_link):
            message = Message(domain=subdomain, url=newest_link, name= p, title = newest_name)
            add_to_queue(json.dumps(message.get()))

if __name__ == "__main__":
    get_newest("http://xembao.vn")
    connection.close()
