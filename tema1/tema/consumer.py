"""
This module represents the Consumer.
Computer Systems Architecture Course
Assignment 1, March 2020
"""

from time import sleep
from threading import Thread


class Consumer(Thread):
    """ Class that represents a consumer. """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor.

        :type carts: List
        :param carts: a list of add and remove operations

        :type marketplace: Marketplace
        :param marketplace: a reference to the marketplace

        :type retry_wait_time: Time
        :param retry_wait_time: the number of seconds that a producer must wait
                                until the Marketplace becomes available

        :type kwargs:
        :param kwargs: other arguments that are passed
                        to the Thread's __init__()
        """

        super(Consumer, self).__init__()
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.cart_id = marketplace.new_cart()


    def run(self):
        for cart in self.carts:
            for action in cart:
                for _ in range(action["quantity"]):
                    if action["type"] == "add":
                        retval = self.marketplace.add_to_cart(self.cart_id, action["product"])
                        while not retval:
                            sleep(self.wait_time)
                            retval = self.marketplace.add_to_cart(self.cart_id, action["product"])
                    elif action["type"] == "remove":
                        self.marketplace.remove_from_cart(self.cart_id, action["product"])

        self.marketplace.place_order(self.cart_id)
