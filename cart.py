import pandas as pd
import numpy as np

#use data from all_item
class Cart:
    def __init__(self, item_list: dict):
        self.item_list = item_list
        self.total_price = 0
        self.cart = {}

    def addItem(self, item: str, amount = 1):
        if item in self.item_list.keys():
            price = self.item_list[item]
            params = np.array([price, amount])
            if item in self.cart.keys():
                self.cart[item] = self.cart[item] + params
                self.total_price += price
            else: 
                self.cart.update({item :params})
                self.total_price += price
            print('Item Added')     
        else:
            print('Invalid item')


    def removeItem(self, item: str, amount = 1):
        if item in self.item_list.keys():
            price = self.item_list[item]
            params = np.array([price, amount])
            if item in self.cart.keys() and self.cart[item][1] != 0:
                self.cart[item] = self.cart[item] - params
                self.total_price -= price
                print('Item Removed') 
            else:
                print('Item is not in the cart')
            
        else:
            print('Invalid item')

    def printCart(self):
        df = pd.DataFrame.from_dict(self.cart, orient='index', columns = ['price', 'amount'])
        print(df)
        print('Total Price: {} Baht'.format(self.total_price))
        print('---------------------------------')