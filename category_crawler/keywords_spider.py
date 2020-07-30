import scrapy

from utils import Articles


class CategorySpider(scrapy.Spider):
    name = 'guardian_category_spider'
    article = Articles('~/Projects/masterthesis/data/article_meta_test_val_train.csv')
    start_urls = article.get_start_urls()
    custom_settings = {
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 3
    }

    def parse(self, response):
        keywords = response.xpath("//meta[@name='keywords']/@content")[0].extract()
        title = response.xpath("//meta[@property='og:title']/@content")[0].extract()

        yield {
            'id': self.article.get_article_id(response.url),
            'url': response.url,
            'title': title,
            'keywords': keywords
        }
