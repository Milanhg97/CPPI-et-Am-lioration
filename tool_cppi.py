import numpy as np
import pandas as pd
from typing import Union
import os
import openpyxl as xls
from datetime import datetime

def investment_period(df:pd.DataFrame, y0:Union[int,str]=2000, horizon:int=5, alldata:bool=False):
    """
    parameters:
        - df : The dataframe
        - y0 : The year, if str, or the date in which the investor start to invest
        - horizon : the period of investment.
        - alldata : a boolean variable.
    
    This function is used to get the dataset during an "investment period", denoted by IP.
        - If alldata is True, the IP start from y0 to the end of the dataset
        - If alldata is False, the IP start from y0 to the y0 to y0+horizon-1
        - If y0's type is int, we include the last day of the previous year into the dataset.
    """
    if (alldata == False):
        if type(y0)==int:
            # Only take the last day of past year
            table_investment_JN = pd.DataFrame(df[str(y0-1):str(y0-1)].iloc[-1]).T 
            # The period of investment
            table_investment_J = pd.DataFrame(df[str(y0):str(y0+horizon-1)])
            # concatenate
            table_investment = pd.concat([table_investment_JN, table_investment_J])
        elif type(y0)==str:
            yyyy = int(y0[:4]) # Give the 4 first letters from yyyy-mm-dd
            mm_dd = y0[4:] # Give the remaining -mm-dd
            yyyh_mm_dd = str(yyyy+horizon-1) + mm_dd
            table_investment = pd.DataFrame(df[y0:yyyh_mm_dd])
    else:
        if type(y0)==int:
            # Only take the last day of past year
            table_investment_JN = pd.DataFrame(df[str(y0-1):str(y0-1)].iloc[-1]).T
            table_investment_J = pd.DataFrame(df[str(y0):])
            table_investment = pd.concat([table_investment_JN, table_investment_J])
        elif type(y0)==str:
            table_investment = pd.DataFrame(df[y0:])        
    return table_investment

def returns(df:pd.DataFrame, initial_investment:float=100.0, y0:Union[int,str]=2000,
           horizon:int=5, alldata:bool=False):
    """
    Parameters:
        - df : The DataFrame
        - initial_investment : The initial value that the client deposits 
        - y0 : The moment (year or day) when the client start to invest
        - horizon : The horizon of investment, by defaul 5 years
        - alldata : 
            - if True : Return the whole dataset from y0 to the end
            - if False : Return the y0 + horizon -1 results
    
    This function allows to compute at first, the returns of the risk asset, from y0 to y0+horizon-1 (if alldata = False)
    or from y0 to the end (if alldata = True). Those return will be apply to the initial amount the client choose to invest
    at beginning
    """
    data = investment_period(df, y0, horizon, alldata)
    S0 = float(data.iloc[0])
    stock_returns = [(float(data.iloc[x])/S0 -1) for x in range(len(data)) ]
    stock_returns = np.array(stock_returns).reshape(-1,1)
    data["Risk return"] = initial_investment * (1 + stock_returns)
    data = data.drop(data.columns[0],axis = 1)
    return data

def monetarize(df:pd.DataFrame, initial_investment:float=100.0, y0:Union[int,str]=2000,
           horizon:int=5, alldata:bool=False):
    """
    parameters:
        - df : The dataset
        - initial_investment : The initial value that the client deposits 
        - y0 : The moment (year or day) when the client start to invest
        - horizon : The horizon of investment, by defaul 5 years
        - alldata : 
            - if True : Return the whole dataset from y0 to the end
            - if False : Return the y0 + horizon -1 results
            
    This function allows to compute the investment to the safe asset (bond), by starting with the amount the investor
    paid.
    """
    data = investment_period(df, y0, horizon, alldata)
    bond_rate = [x for x in data.iloc[:,0]]
    bond_rate = np.array(bond_rate).reshape(-1,1)
    bond = []
    compteur = 0
    for i in bond_rate:
        if compteur == 0:
            bond.append(initial_investment)
        else:
            investment = float(bond[compteur-1] * (1 + i/(365*100)))
            bond.append(investment)
        compteur += 1
    data["RiskFree return"] = bond
    data = data.drop(data.columns[0], axis = 1)
    return data

def createxls(df:pd.DataFrame, opt:bool=False):
    # XLS futures names 
    now = datetime.now()
    yyyy = str(now.year)
    mm = str(now.month).zfill(2)
    dd = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    second = str(now.second).zfill(2)
    name = f"DataFrame_{yyyy}_{mm}_{dd}_{hour}_{minute}_{second}"
    # Create excel
    work = xls.Workbook()
    work.active #take the first sheet by default
    # Path
    path = os.getcwd()
    full_name = path + "\\01. Excel Data\\" + name +".xlsx"
    #Save :
    work.save(full_name)
    work.close()
    # Updated 
    df.to_excel(full_name, index = opt)
    print(f"The following excel is successfuly registered : {name}")
    
def bond(df:pd.DataFrame, r:float=2, naming:str="Fixed")->pd.DataFrame:
    """
    Parameters:
        - df: DataFrame
        - r: float
        - naming: str
    Returns:
        - pd.DataFrame
    
    """
    ind = df.index
    rt = [r for i in range(len(ind))]
    df = pd.DataFrame({naming:rt}, index = ind)
    return df

class Information:
    def __init__(self, dfrisk:pd.DataFrame, dfrate:pd.DataFrame, 
                 initial_investment:float=100.0, garant:float=1.0, y0:str="2000",
                 horizon:int=5, multiplier:float=2, activation:list=[3,6,9,12]):
        
        # Instance our 8 parameters
        self.dfrisk = pd.DataFrame(dfrisk.copy())
        self.dfrate = pd.DataFrame(dfrate.copy())
        self.initial_investment = initial_investment
        self.garant = garant
        self.y0 = y0
        self.horizon = horizon
        self.multiplier = multiplier
        self.activation = activation
        
        # Create others instances
        self.nb_days = len(self.dfrisk)
        self.nb_period_year = self.nb_days / self.horizon
        self.multip = multiplier
        
        # Date and Horizon's format : yyyy(-mm-dd) and (yyyy+h)(-mm-dd)
        self.yyyy = int(self.y0[:4])
        self.yyyh = str(self.yyyy + self.horizon-1)
        if len(self.y0)==4:
            self.inception = str(self.yyyy)
            self.end = self.yyyh
            self.onlyear = True
        elif len(self.y0)==7:
            self.mm = str(self.y0[5:7])
            self.inception = str(self.yyyy) + "-" + self.mm
            self.end = str(self.yyyh) + "-" + self.mm
            self.onlyear = False
        elif len(self.y0) == 10:
            self.mm = str(self.y0[5:7])
            self.dd = str(self.y0[8:10])
            self.inception = str(self.yyyy) + "-" + str(self.mm) + "-" + str(self.dd)
            self.end = str(self.yyyh) + "-" + str(self.mm) + "-" + str(self.dd)
            self.onlyear = False
    
    def period(self, tab:pd.DataFrame)->pd.DataFrame:
        """
        Parameter:
            - opt: tab
        Returns:
            - pd.DataFrame
        Description:
            This method allows to return tab from the inception to the end of the investment.
                - y0 is the string's parameters which is the investment's inception date. It could be by type 
                yyyy or yyyy-mm or yyyy-mm-dd.
                - horizon, which is type int, will be the horizon of investment. 
        
        """
        if self.onlyear:
            # Last data of previous year
            tab_JN = pd.DataFrame(tab[str(self.yyyy-1):str(self.yyyy-1)].iloc[-1]).T
            # Data from y0 to y0+horizon-1
            tab_J = tab[str(self.yyyy):self.yyyh]
            # Concatenate
            table = pd.concat([tab_JN, tab_J])
        else:
            table = pd.DataFrame(tab[self.inception:self.end])
        return table
    
    def check_period(self, opt:bool=False)->pd.DataFrame:
        """
        Parameter:
            - opt: bool
        Returns:
            - pd.DataFrame
        Description:
            This method allows to give a screen about our data:
                - If opt is True: It returns the screen about risky asset
                - If opt is False: It returns the screen about free-risk asset
        """
        if opt:
            tab = pd.DataFrame(self.dfrisk.copy())
        else:
            tab = pd.DataFrame(self.dfrate.copy())
        
        table = self.period(tab)
        return table
    
    def zooming(self, data:pd.DataFrame, inception:str, end:str)->pd.DataFrame:
        """
        Parameters:
            - data: pd.DataFrame,
            - inception: str,
            - end: str
        Returns:
            - pd.DataFrame
        Description:
            This method is used to check data into "inception" to "end"
        """
        return data.loc[inception:end]
    
    def pct_returns(self, value:float=100, opt:bool=True)->pd.DataFrame:
        """
        Description:
            This method allows to get a DataFrame of the risky asset's returns in percentage. 
        """
        if opt:
            tab = self.check_period(True)
        else:
            tab = self.dfrisk
        S0 = float(tab.iloc[0,0])
        stock_returns = [(float(tab.iloc[x])/S0 -1) for x in range(len(tab))]
        stock_returns = np.array(stock_returns).reshape(-1,1)
        tab["Risky return"] = value * (1 + stock_returns)
        tab = tab.drop(tab.columns[0],axis = 1)
        return tab
    
    def returns(self, col_name:str="Risk return", opt:bool=True, typing:bool=True)->pd.DataFrame:
        """
        Parameters:
            col_name:str
            opt:bool
            typing:bool
        Returns:
            pd.DataFrame
        Description:
            This method allows to get a DataFrame of the risky asset's returns.
            if:
                - opt is True, get the returns from the inception to the end of the INVESTMENT,
                - opt is False, get the returns to the whole data of risky asset
                - typing is True, get the net returns in %, i.e, (S[t]/S[t-1])-1
                - typing is False, get the log-returns in %, i.e, log(S[t]/S[t-1])
        """
        if opt:
            tab = self.check_period(True)
        else:
            tab = pd.DataFrame(self.dfrisk.copy())
        stock_returns = [1.0] #100%
        for i in range(len(tab)):
            if i == 0 :
                pass
            else:
                stock_returns.append(float(tab.iloc[i,0])/float(tab.iloc[i-1,0]))
        stock_returns = np.array(stock_returns).reshape(-1,1)
        if typing:
            tab[col_name] = (stock_returns - 1) * 100
        else:
            tab[col_name] = np.log(stock_returns) * 100
        tab = tab.drop(tab.columns[0],axis = 1)
        return tab
    
    def rolling_volatility(self, nb_days:int=90, amplitude:float=np.sqrt(256)*100,opt:bool=True)->pd.DataFrame:
        """
        Parameters:
            nb_days: int
            amplitude: float
            opt: bool
        Return:
            pd.DataFrame
        Description:
            This method allows to compute the volatility with a windows of nb_days days from the inception of
            the risk dataframe to the end, by using the method "returns", in which it computes the daily returns 
            and we can apply a window to compute the standard deviation. Once computed, it returns data from
            the inception of the investment to the end of the investment.
            The parameter "amplitude" allows adjusting the volatility to a larger or a smaller scale.
            When the parameter amplitude equals to np.sqrt(256), it computes the annualized volatility.
        """
        tab = self.returns(opt=False,typing=False) # Log-returns
        vol_df = (tab.rolling(window = nb_days).std()) * amplitude
        
        if opt:
            dff = self.period(vol_df)
        else:
            dff = vol_df
        ind = dff.index
        vol_list = [vol for vol in dff.iloc[:,0]]
        vol_tab = pd.DataFrame({"Volatility":vol_list}, index = ind)
        return vol_tab
    
    def constant_information(self)->pd.DataFrame:
        """
        Description:
            This method allows to get a DataFrame which show logically known data:
                - Risk return,
                - Rate,
                - Time (int)
        """
        # Create constant information : Risk return, Rate
        risk0 = self.returns("Risk return %")
        norisk0 = self.check_period(False)
        tab_vol = self.rolling_volatility(nb_days=90, amplitude=np.sqrt(256),opt=True)
        t = pd.DataFrame({"t":range(len(risk0))}, index = risk0.index)
        table = pd.concat([t, risk0, norisk0, tab_vol], axis = 1)
        return table
    
class Central_CPPI(Information):
    def __init__(self, dfrisk:pd.DataFrame, dfrate:pd.DataFrame, decision:pd.DataFrame,
                 initial_investment:float=100.0, garant:float=1.0, y0:str="2000",
                 horizon:int=5, multiplier:float=2, activation:list=[3,6,9,12]):
        super().__init__(dfrisk, dfrate, initial_investment, garant, y0, horizon, multiplier, activation)
        self.decision = pd.DataFrame(decision.copy())
    
    def date_index(self)->list:
        """
        This method allows getting a list of dates, in which the fund updates the floor, periodically.
        For example, if self.activation = [3,6,9,12], it means that each quater, the floor will be 
        updated such that it reaches the level of self.garant * the previous VL.
        """
        table = self.constant_information()#[0]
        # Date
        year_begin = int(str(table.index[0])[:4])
        year_end = int(str(table.index[-1])[:4])
        month_begin = int(str(table.index[0])[5:7])
        
        # Initialize
        monthly = [month_index for month_index in self.activation if month_index > month_begin]
        trim = [str(table.index[0])] # Last day of past year
        
        # Looping
        for year in range(year_begin,year_end+1):
            if year==year_begin:
                for month in monthly:
                    if len(str(month))==1:
                        mm = "0" + str(month)
                    else:
                        mm = str(month)
                    yyyymm = str(year) + "-" + mm
                    trim.append(str(table.loc[yyyymm].index[0]))
            else:
                for month in self.activation:
                    yyyy = str(year)
                    if len(str(month))==1:
                        mm = "0" + str(month)
                    else:
                        mm = str(month)
                    yyyymm = yyyy + "-" + mm
                    trim.append(str(table.loc[yyyymm].index[0]))
        last_part = table.loc[trim[-1]:].index[-1]
        trim.append(str(last_part))
        self.end = year_end
        return trim
    
    def check_screen(self, data:pd.DataFrame, inception:str, end:str)->pd.DataFrame:
        """
        Parameters:
            - data: pd.DataFrame,
            - inception: str,
            - end: str
        Description:
            This method allows to get a DataFrame, from "inception" to "end"
        """
        return data.loc[inception:end]
    
    def garantide(self, value:float, opt:bool=True)->float:
        """
        Parameters:
            - value: float,
            - opt: bool
        Description:
            This method allows to get the guaranteed value (GV).
            if:
                - opt is True: the GV is simply the value that we enter in the parameter
                - opt is False: the GV will be the "garant" parameters of the initial investment.
        """
        if opt:
            garantia = value
        else:
            garantia = self.garant * self.initial_investment
        return garantia
    
    def garantie(self, value:float, inception:str, end:str, opt:bool = True)->pd.DataFrame:
        """
        Parameters:
            - value: float,
            - inception:str,
            - end: str,
            - opt: bool
        Description:
            This method allows to add a column of the guaranteed value. These value will firstly be fixed.
            In another method, the guaranteed value will be discounted and therefore will become the "floor"
        """
        info = self.constant_information()
        table = self.check_screen(info, inception, end)
        val = self.garantide(value,opt)
        table2 = pd.DataFrame(table.copy())
        table2.loc[:,"Garantie"] = val
        return table2
    
    def flooring(self, value:float, inception:str, end:str, opt:bool = True)->pd.DataFrame:
        """
        Parameters:
            - value: float,
            - inception:str,
            - end: str,
            - opt: bool
        Description:
            This method allows to compute the floor, which varies everyday. In the C.P.P.I model, the floor 
            is important because the VL should not reach it in downside. The floor's formula depends on the 
            guaranteed capital (for instance, 100% of the initial investment) in which we should discounted it.
        """
        table = self.garantie(value, inception, end, opt)
        # Initialization
        r = list(table.iloc[:,2]) # norisk
        t = list(table.iloc[:,0]) # t
        garanteed_capital = list(table.iloc[:,4]) # Garantie
        planche = np.zeros((len(table),1))
        nb_period = self.nb_days/self.horizon # Business day, by mean
        
        for k in range(len(table)):
            planche[k] = garanteed_capital[k] / ((1 + (r[k]/100))**(self.horizon - t[k]/nb_period))
        
        table["Floor"] = planche
        
        return table
    
    def liquidevalue(self, value:float, opt:bool)->float:
        """
        Parameters:
            - value: float,
            - opt: bool
        Description:
            This method returns the:
                - either the value that we put as parameter, if opt = True
                - or the initial investment if opt = False
        """
        if opt:
            VL = value
        else:
            VL = self.initial_investment
        return VL
    
    def vol_multiplier(self, vol:float)->int:
        """
        Parameters:
            - vol: float,
        Description:
            This method returns the multiplier according to the volatility
        """
        if (vol<10) and (vol>0):
            return 7
        elif vol < 20:
            return 4
        elif vol < 30:
            return 3
        elif vol < 40:
            return 2
        elif vol < 50:
            return 1
        else:
            return 0
    
    def central_decision_multiplier(self, mtp:float, cond:int):
        """
        Parameters:
            - mtp : float
            - cond : int
        Returns:
            - float
        Description :
            This methods will be used to change the multiplier according to 
            the central bank decision (that we get thanks to the NLP's algorithm)

        """
        if (cond==-1):
            ########
            # If we predict that the rate will growing DOWN, it means that, it is less interesting to 
            # invest to Bond (according to our result). Hence, we give more importance to the multiplier, 
            # since it allows us to invest more to risky asset
            ########
            val = mtp * (1 + 50/100)
        elif (cond==0):
            ########
            # The prediction says that the rate will neither growing up nor down significantly or
            # the Central Bank doesn't give any decision publicly. Therefore, the multiplier stay the same
            ########
            val = mtp
        elif (cond==1):
            ########
            # If we predict that the rate will growing UP, it means that, it is more interesting to 
            # invest to Bond (according to our result). Hence, we give less importance to the multiplier,
            # since it allows us to invest more risky asset 
            ########
            val = mtp * (1 - 50/100)
        return val
    
    def principale(self, value_garant:float, value_invest:float,vol_opt:bool, inception:str, end:str, 
                   opt:bool = True, multi_opt:bool=True)->pd.DataFrame:
        """
        Parameters:
            - value_garant: float,
            - value_invest: float,
            - inception: str,
            - end: str,
            - opt: bool
            - multi_opt:bool
        Description:
            This method is the key one of this class. It returns every informations of the table.
            By "informations", we meant 
                - known value (time, risk data, free risk data, floor)
                - unknown value (cushion, multiple, data before and after adjustment etc...)
        """
        # Known data
        table = self.ready_table(value_garant, inception, end, opt)
        risk = list(table.iloc[:,1])
        norisk = list(table.iloc[:,2])
        volatility = list(table.iloc[:,3])
        plancher = list(table.iloc[:,5])
        bank = list(table.iloc[:,6])

        # Initialization:

        val = [value_invest] #1
        cushion = [val[0] - plancher[0]] #2

        if multi_opt:
            if vol_opt:
                multip = self.vol_multiplier(volatility[0])
                multipl = self.central_decision_multiplier(multip, bank[0])
            else:
                multip = self.multiplier
                multipl = self.central_decision_multiplier(multip, bank[0])
        else:
            if vol_opt:
                multipl = self.vol_multiplier(volatility[0])
            else:
                multipl = self.multiplier
        self.multiplier = multipl
        
        if (cushion[0] * multipl <= self.initial_investment):
            ########
            # If the C[0] * m <= VL[0], it means that we can buy Cushion*m€ of the risky asset
            ########
            risky_before_adj = [cushion[0] * multipl] #3
            multiplier_adj = [multipl] #8
        else:
            ########
            # If C[0] * m > VL[0], i.e we buy a higher amount of risky asset such that the
            # initial amount couldn't handle it. Therefore, we only can initialize the 
            # risky asset with the whole amount of the initial investment, and 0€ of B[t]
            ########
            risky_before_adj = [val[0]]
            multiplier_adj = [val[0]/cushion[0]] #8
        norisk_before_adj = [val[0] - risky_before_adj[0]] #4
        multiplier_before_adj = [risky_before_adj[0] / cushion[0]] #5
        risky_adj = [risky_before_adj[0]] #6
        norisk_adj = [norisk_before_adj[0]] #7
        flow_adj = [0] #9     
        
        # Looping
        for t in range(len(table)):
            if t == 0:
                pass
            else:
                risky_before_adj.append(risky_adj[t-1] * (1 + risk[t]/100)) 
                norisk_before_adj.append(norisk_adj[t-1] * (1 + norisk[t]/(100*self.nb_period_year)))   
                val.append(risky_before_adj[t] + norisk_before_adj[t])

                if multi_opt:
                    if vol_opt:
                        multip = self.vol_multiplier(volatility[t])
                        multipl = self.central_decision_multiplier(multip, bank[t])
                    else:
                        multip = self.multiplier
                        multipl = self.central_decision_multiplier(multip, bank[t])
                else:
                    if vol_opt:
                        multipl = self.vol_multiplier(volatility[t])
                    else:
                        multipl = self.multiplier 
                self.multiplier = multipl
                
                
                ######### 
                # if flow_adjustment > 0 : We have to buy the risky asset (financed by the safe asset),
                # and reciprocally, i.e if flow_adjustment < 0, we have to buy the safe asset (financed by the risky asset)
                #########
                if (val[t-1] >= plancher[t-1]): 
                    ###########
                    # If liquidative value is greater than the floor, i.e, the Cushion is positive
                    ###########
                    flow = 0
                    cushion.append(val[t] - plancher[t])
                    multiplier_before_adj.append(risky_before_adj[t]/cushion[t])

                    if (multipl * cushion[t] - risky_before_adj[t] < norisk_before_adj[t]):
                        ###########
                        # If m*C[t] - E[t] = Flow[t] < B[t], which means that we can fully convert Bonds 
                        # into Risky asset if we have to buy it
                        ###########
                        flow = multipl * cushion[t] - risky_before_adj[t]
                    else: 
                        ###########
                        # Else, and if B[t] > 0, we only can (partially) buy the risky asset E[t] 
                        # with an amount of B[t]€ (which is not sufficiency enough)
                        ###########
                        if (norisk_before_adj[t] >= 0):
                            flow = norisk_before_adj[t]
                else: 
                    ###########
                    # If the Liquidative Value is lower than the Floor, we decide to totally monetarize, 
                    # i.e, 100% in Bond, by taking Flow[t] = E[t]
                    ###########
                    flow = -risky_before_adj[t]
                    cushion.append(0)
                    multiplier_before_adj.append(0)
                
                if (risky_before_adj[t] + flow > 0): 
                    ###########
                    # If the risky asset E[t] is high enough 
                    ###########
                    flow_adj.append(flow)
                else: 
                    ###########
                    # If the risky asset isn't high enough: 
                    # We only can sell the amount of E[t]€ to get some B[t]
                    ###########
                    flow_adj.append(-risky_before_adj[t])
                    
                risky_adj.append(risky_before_adj[t] + flow_adj[t])
                norisk_adj.append(norisk_before_adj[t] - flow_adj[t])
                if (cushion[t]!=0):
                    multiplier_adj.append(risky_adj[t]/cushion[t])
                else:
                    multiplier_adj.append(multipl)
                

        # Value attribution
        table["NAV"] = val
        table["Risk"] = risky_before_adj
        table["Safe asset"] = norisk_before_adj
        table["Cushion"] = cushion
        table["Multiplier before adj."] = multiplier_before_adj
        table["Flow Adjustment"] = flow_adj
        table["Risk with adj."] = risky_adj
        table["Safe Asset with adj."] = norisk_adj
        table["Multiplier with adj."] = multiplier_adj
        table["Volatility"] = volatility       
        
        return table

    def combined(self, vol_opt:bool=False, multi_opt:bool=True)->list:
        """
        Parameters:
            - vol_opt:bool
            - multi_opt:bool
        Return:
            A list of pd.DataFrame
        Description:
            This method allows to combined each crucial information, from the method self.principal(...)
            inside a list. In other word, we sub-divise the data into many part, related to the parameter
            self.activation and the method self.date_index(), such that the guaranteed floor is regularly
            updated, according to the client's preference.
        """
        # Preparation
        data_ind = self.date_index()
        list_data = []
        
        for i in range(len(data_ind)-1):
            if i == 0:
                garant_val = self.garantide(self.garant * self.initial_investment, False) # Not obligation to enter an accurate Value, bcse False.
                data = self.principale(garant_val, self.initial_investment, vol_opt, data_ind[i], data_ind[i+1], False, multi_opt)
                list_data.append(data[:-1])
            else:
                last_data = list_data[-1]
                last_VL = float(last_data.iloc[-1,7])
                last_VL_gar = last_VL * self.garant
                data = self.principale(last_VL_gar, last_VL, vol_opt, data_ind[i], data_ind[i+1], True, multi_opt)
                if i == len(data_ind)-2:
                    list_data.append(data)
                else:
                    list_data.append(data[:-1])
        return list_data    
    
    def cppi(self, vol_opt:bool=False, multi_opt:bool=False)->pd.DataFrame:
        """
        Parameters:
            - vol_opt:bool
            - multi_opt
        Return:
            pd.DataFrame
        Description:
            Use the list of DataFrame from the method self.combined() and concatenate it with axis = 0
        """
        data_list = self.combined(vol_opt, multi_opt)
        data = pd.concat(data_list, axis = 0)
        data = data.drop(data.columns[0], axis = 1) #t
        return data
    
    def crucial(self, opt:bool=False, vol_opt:bool=False, multi_opt:bool=False)->pd.DataFrame:
        """
        Parameter:
            - opt:bool
            - vol_opt:bool
            - multi_opt:bool
        Return:
            pd.DataFrame
        Description:
            Returns only :
            - The Ricky asset returns,
            - the Liquidative Value, 
            - the Cushion
            - the Volatility
        """
        table = self.cppi(vol_opt, multi_opt)
        risky_val = table["Risk"].iloc[0]
        if opt:
            risk = self.pct_returns(risky_val)
        else:
            risk = self.pct_returns(self.initial_investment)
        tableVL = pd.DataFrame(table["NAV"])
        tableCoussin = pd.DataFrame(table["Cushion"])
        tablePlancher = pd.DataFrame(table["Floor"])
        tableVol = pd.DataFrame(table["Volatility"])
        df = pd.concat([risk, tableVL, tableCoussin, tablePlancher, tableVol], axis = 1)
        df = df.dropna(axis = 0)
        return df
    
    def ready_table(self, value:float, inception:str, end:str, opt:bool = True)->pd.DataFrame:
        """
        Parameters:
            - value: float,
            - inception:str,
            - end: str,
            - opt: bool
        Description:
            Take the output of the method Flooring, and add the Centrale Bank Decision
        """
        table = self.flooring(value, inception, end, opt)
        rate_pred = pd.DataFrame(self.decision.copy())
        df_pred = self.zooming(self.period(rate_pred), inception, end)
        #pd.DataFrame(taux[self.inception:self.end])
        liste = [element for element in df_pred.iloc[:,0]]
        table["Central Bank"] = liste
        return table
