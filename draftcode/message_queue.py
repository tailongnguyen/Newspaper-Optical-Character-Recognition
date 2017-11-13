class Message():
    def __init__(self, domain="", url="", name="", title = "", link_image="", path = "", \
                Datetime_crawl = None, Datetime_extract = None, text = ""):
        self.domain = domain
        self.url = url
        self.name = name
        self.title = title
        self.link_image = link_image
        self.path = path
        self.Datetime_crawl = Datetime_crawl
        self.Datetime_extract = Datetime_extract
        self.text = text
    
    def get(self):
        message = {
            "domain":self.domain,
            "url":self.url,
            "name":self.name,
            "title":self.title,
            "link_image":self.link_image,
            "path":self.path,
            "Datetime_crawl":self.Datetime_crawl,
            "Datetime_extract":self.Datetime_extract,
            "text":self.text
        }
        return message