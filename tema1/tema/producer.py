"""
This module represents the Producer.
Computer Systems Architecture Course
Assignment 1
March 2020
"""

from time import sleep
from threading import Thread


class Producer(Thread):
    """
    Class that represents a producer.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor.

        @type products: List()
        @param products: a list of products that the producer will produce

        @type marketplace: Marketplace
        @param marketplace: a reference to the marketplace

        @type republish_wait_time: Time
        @param republish_wait_time: the number of seconds that a producer
                                    must wait until the marketplace
                                    becomes available

        @type kwargs:
        @param kwargs: other arguments that are passed
                        to the Thread's __init__()
        """

        super(Producer, self).__init__(daemon=True)
        self.products = products
        self.marketplace = marketplace
        self.wait_time = republish_wait_time
        self.producer_id = marketplace.register_producer()


    def run(self):
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    retval = False
                    while not retval:
                        retval = self.marketplace.publish(self.producer_id, product[0])
                        sleep(product[2])
