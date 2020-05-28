"""
This module represents the Marketplace.
Computer Systems Architecture Course
Assignment 1, March 2020
"""

# pylint: disable=too-many-instance-attributes
# Sunt necesare toate cele 8 atribute din clasă

from threading import Semaphore


class Marketplace:

    """
    Class that represents the Marketplace.
    It's the central part of the implementation.
    The producers and consumers use its methods concurrently.
    """

    def __init__(self, queue_size_per_producer):
        """
        Constructor

        :type queue_size_per_producer: Int
        :param queue_size_per_producer: the maximum size of a queue
                                        associated with each producer
        """

        self.new_producer_id = 0
        self.new_cart_id = 0
        self.carts = {}
        self.products = {}
        self.producers = {0:0}
        self.max_size = queue_size_per_producer
        self.sem_producer = Semaphore(2000)
        self.sem_consumer = Semaphore(0)


    def register_producer(self):
        """ Returns an id for the producer that calls this. """

        self.new_producer_id = self.new_producer_id + 1
        self.producers[self.new_producer_id] = 0
        return self.new_producer_id


    def publish(self, producer_id, product):
        """
        Adds the product provided by the producer to the marketplace

        :type producer_id: String
        :param producer_id: producer id

        :type product: Product
        :param product: the Product that will be published in the Marketplace

        :returns True or False. If the caller receives False,
                                it should wait and then try again.
        """

        if self.producers[producer_id] < self.max_size:
            self.sem_producer.acquire()

            if not product in self.products:
                self.products[product] = [producer_id]
            else:
                self.products[product].append(producer_id)

            self.producers[producer_id] = self.producers[producer_id] + 1
            self.sem_consumer.release()
            return True

        return False


    def new_cart(self):
        """
        Creates a new cart for the consumer

        :returns an int representing the cart_id
        """

        self.new_cart_id = self.new_cart_id + 1
        self.carts[self.new_cart_id] = []
        return self.new_cart_id


    def add_to_cart(self, cart_id, product):
        """
        Adds a product to the given cart. The method returns

        :type cart_id: Int
        :param cart_id: id cart

        :type product: Product
        :param product: the product to add to cart

        :returns True or False. If the caller receives False,
                                it should wait and then try again
        """

        self.sem_consumer.acquire()

        if product in self.products:
            if self.products[product]:
                self.carts[cart_id].append(product)
                producer_id = self.products[product].pop()
                self.producers[producer_id] = self.producers[producer_id] - 1
                self.sem_producer.release()
                return True

        self.sem_producer.release()
        self.sem_consumer.release()
        return False


    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from cart.

        :type cart_id: Int
        :param cart_id: id cart

        :type product: Product
        :param product: the product to remove from cart
        """

        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)
            self.publish(0, product)


    def place_order(self, cart_id):
        """
        Return a list with all the products in the cart.

        :type cart_id: Int
        :param cart_id: id cart
        """

        for product in self.carts[cart_id]:
            print(f'cons{cart_id} bought {product}')
