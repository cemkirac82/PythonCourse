import numpy as np
class Portfolio(object):
    
        def __init__(self,owner='CEM KIRAC'):
            self.owner = owner
            self.cash  = 0.0
            self.stocks = {}
            self.mutualfunds = {}
            self.actionslog = ['\nAUDIT LOG IN CHRONOLOGICAL ORDER\n--------------------------']
        
        def addCash(self,amount):
            self.cash+=amount
            self.actionslog.append('Cash Added $'+str(amount))
        
        def withdrawCash(self,amount):
            self.cash-=amount
            self.actionslog.append('Cash Withdrew $'+str(amount))
            
        def history(self):
            for x in self.actionslog:
                print(x)
                
        def buyStock(self, count, any_stock_instance):
            self.cash-=count*any_stock_instance.buyprice
            any_stock_instance.buy(count)
            self.stocks[any_stock_instance.stockname]=[round(any_stock_instance.count,3)]
            self.actionslog.append('Bought Stocks'+str((any_stock_instance.stockname,count)) + ' Cash Cost Total $'+str(count*any_stock_instance.buyprice))

        def sellStock(self, any_stock_instance,count):
            self.cash+=count*any_stock_instance.sellprice
            any_stock_instance.sell(count)
            self.stocks[any_stock_instance.stockname]=[any_stock_instance.count]
            self.actionslog.append('Sold Stocks'+str((any_stock_instance.stockname,count)) + ' Cash Income Total $'+str(count*any_stock_instance.sellprice))

        def buyMutualFund(self, count, any_fund_instance):
            self.cash-=count*any_fund_instance.buyprice
            any_fund_instance.buy(count)
            self.mutualfunds[any_fund_instance.name]=[any_fund_instance.count]
            self.actionslog.append('Bought MutualFunds'+str((any_fund_instance.name,count)) + ' Cash Cost Total $'+str(count*any_fund_instance.buyprice))

        def sellMutualFund(self,  any_fund_instance,count):
            self.cash+=count*any_fund_instance.sellprice
            any_fund_instance.sell(count)
            self.mutualfunds[any_fund_instance.name]=[round(any_fund_instance.count,3)]
            self.actionslog.append('Sold MutualFunds'+str((any_fund_instance.name,count)) + ' Cash Income Total $'+str(count*any_fund_instance.sellprice))

        def __str__(self):
            b='\nCEM KIRAC INVESTMENT PORTFOLIO\n--------------------------\ncash:         $'+str(round(self.cash,2))+'\n'
            for k, v in self.stocks.items():
                if list(self.stocks.keys())[0]==k:
                    a='stock:        '+str(k)+' '+str(v)+'\n'
                else:
                    a='              '+str(k)+' '+str(v)+'\n'
                b+=a
            for k, v in self.mutualfunds.items():
                if list(self.mutualfunds.keys())[0]==k:
                    a='mutual funds: '+str(k)+' '+str(v)+'\n'
                else:
                    a='              '+str(k)+' '+str(v)+'\n'
                b+=a
            return b
# =============================================================================
#             print(str(self.stocks))
#             print(str(self.mutualfunds))
# =============================================================================
            
class Stock(object):
    
    def __init__(self,buyprice,stockname):
            self.stockname = stockname
            self.buyprice=buyprice
            self.sellprice=self.buyprice*np.random.uniform(1/2,3/2)
            self.count=0
            
    def buy(self,number_of_stocks):
            self.count+=round(number_of_stocks,3)

    def sell(self,number_of_stocks):
            self.count-=round(number_of_stocks,3)
            
class MutualFund(object):
    def __init__(self,name):
            self.name = name
            self.buyprice= 1
            self.sellprice=self.buyprice*np.random.uniform(0.9,1.2)
            self.count=0
            
    def buy(self,number_of_funds):
            self.count+=round(number_of_funds,3)

    def sell(self,number_of_funds):
            self.count-=round(number_of_funds,3)    
            
portfolio = Portfolio() #Creates a new portfolio
portfolio.addCash(300.50) #Adds cash to the portfolio
s = Stock(20, "HFH") #Create Stock with price 20 and symbol "HFH"
portfolio.buyStock(5, s) #Buys 5 shares of stock s
mf1 = MutualFund("BRT") #Create MF with symbol "BRT"
mf2 = MutualFund("GHT") #Create MF with symbol "GHT"
portfolio.buyMutualFund(10.3, mf1) #Buys 10.3 shares of "BRT"
portfolio.buyMutualFund(2, mf2) #Buys 2 shares of "GHT"
print(portfolio) #Prints portfolio
portfolio.sellMutualFund(mf1, 3) #Sells 3 shares of BRT
portfolio.sellStock(s, 1) #Sells 1 share of HFH
portfolio.withdrawCash(50) #Removes $50
portfolio.history() #Prints a list of all transactions





