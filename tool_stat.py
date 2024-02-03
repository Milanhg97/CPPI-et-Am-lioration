import numpy as np
import pandas as pd   


class StatCPPI:
    def __init__(self, df:pd.DataFrame, dfrisk:pd.DataFrame, dfriskless:pd.DataFrame):
        self.df = pd.DataFrame(df.copy())
        self.dfrisk = pd.DataFrame(dfrisk.loc[self.df.index].copy())
        self.dfriskless = pd.DataFrame(dfriskless.loc[self.df.index].copy())
    
    def pct_data(self, data:pd.DataFrame)->pd.DataFrame:
        """
        Parameters:
            - data:pd.DataFrame
        Returns:
            - pd.DataFrame
        Description:
            This method returns the daily variation rate of the DataFrame "data"
        """
        tab = pd.DataFrame(data.copy())
        tab = tab.pct_change().fillna(0)
        return tab
    
    def mean(self, msg:bool=False, opt:bool=False)->float:
        """
        Parameters:
            - msg:bool
            - opt:bool
        Returns:
            - pd.DataFrame
        Description:
            - This method returns the mean of the returns
        """
        if opt:
            tab = pd.DataFrame(self.df.copy())
        else:
            tab = pd.DataFrame(self.dfrisk.copy())
        table = self.pct_data(tab)
        moy = float(table.mean())
        if msg:
            if opt:
                print(f"The Average of the Fund's return is : {round(moy*100,5)}%")
            else:
                print(f"The Average of the risky asset's return is : {round(moy*100,5)}%")
        else:
            pass
        return moy

    def returns(self, data:pd.DataFrame, col_name:str="Risk return", typing:bool=True)->pd.DataFrame:
        """
        Parameters:
            - data:pd.DataFrame
            - col_name:str
            - opt:bool
            - typing:bool
        Returns:
            pd.DataFrame
        Description:
            This method allows to get a DataFrame of the risky asset's returns, with log-returns.
            if:
                - opt is True, get the returns from the inception to the end of the INVESTMENT,
                - opt is False, get the returns to the whole data of risky asset
        """
        tab = pd.DataFrame(data.copy())
        S0 = tab.iloc[0,0]
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
    
    def annualized_volatility_NAV(self, msg:bool=True)->float:
        """
        Parameters:
            - msg:bool
        Returns:
            - float
        Description:
            This method computes the annualized volatility of the Fund
        """
        tab = pd.DataFrame(self.df.copy())
        logreturns = self.returns(data = tab, col_name = "NAV Log-returns", typing=False)
        vol = float(logreturns.std()*np.sqrt(256))
        if msg:
            print(f"The annualized volatility of the strategy is : {round(vol,5)}%")
        else:
            pass
        return vol
    
    def annualized_volatility_risk(self, msg:bool=True)->float:
        """
        Parameters:
            - msg:bool
        Returns:
            - float
        Description:
            This method computes the annualized volatility of the risky asset
        """
        tab = pd.DataFrame(self.dfrisk.copy())
        logreturns = self.returns(data = tab, col_name = "Risk Log-returns", typing=False)
        vol = float(logreturns.std()*np.sqrt(256))
        if msg:
            print(f"The annualized volatility of the risky asset is : {round(vol,5)}%")
        return vol
    
    def variation_coeff(self, msg:bool=False, opt:bool=False)->float:
        """
        Parameters:
            - msg:bool
            - opt:bool
        Returns:
            - float
        Description:
            This method computes the variation's coefficient of the data
        """        
        if opt:
            dff = pd.DataFrame(self.df.copy())
            m = float(dff.mean())
            std = float(dff.std())
        else:
            dff = pd.DataFrame(self.dfrisk.copy())
            m = float(dff.mean())
            std = float(dff.std())
        if msg:
            if opt:
                print(f"The variation coefficient of the Fund is: {std/m}")
            else:
                print(f"The variation coefficient of the risky asset is: {std/m}")
        else:
            pass
        return std/m
    
    def annualized_returns(self, data:pd.DataFrame, msg:bool=False, opt:bool=True):
        """
        Parameters:
            - data:pd.DataFrame
            - msg:bool
            - opt:bool
        Returns:
            - float
        Description:
            This method computes the annualized returns of the "data"
        """
        tab = pd.DataFrame(data.copy())
        initiale = float(tab.iloc[0,0]) 
        finale = float(tab.iloc[-1,0])
        year_initiale = int(str(tab.iloc[0:,0].index[0])[0:4])
        year_finale = int(str(tab.iloc[0:,0].index[-1])[0:4])
        nb_year = year_finale-year_initiale
        rendement_ann = (finale/initiale)**(1/nb_year) -1
        
        if msg == True:
            if opt:
                print(f"The annualized return of the Fund is: {round(rendement_ann*100,5)}%")
            else:
                print(f"The annualized return of the risky asset is: {round(rendement_ann*100,5)}%")
        return rendement_ann
    
    def drawdown(self, data:pd.DataFrame, msg:bool=False,opt:bool=False)->float:
        """
        Parameters:
            - data:pd.DataFrame
            - msg:bool
            - opt:bool
        Returns:
            - float
        Description:
            This method computes the Maximum DrawDown of the "data"
        """
        tab = pd.DataFrame(data.copy())
        tab["DrawDown"] = 0
        n = len(tab)
        vl = [x for x in tab.iloc[:,0]]
        DD = []
        for i in range(len(vl)):
            if i < len(vl)-1:
                current = vl[i]
                draw = (min(vl[i+1:])/current)-1
                DD.append(draw)
        maxdraw = min(DD)

        themax_index = DD.index(maxdraw)
        if msg==True:
            if opt:
                print(f"The max drawdown of the Fund: {round(maxdraw*100,2)}%")
            else:
                print(f"The max drawdown of the risky asset is: {round(maxdraw*100,2)}%")
        return maxdraw
    
    def sharperatio(self, msg:bool=False, opt:bool=False)->float:
        """
        Parameters:
            - msg:bool
            - opt:bool
        Returns:
            - float
        Description:
            This method computes the Sharpe Ratio of:
                - The Fund, if opt is True
                - The risky asset, if opt is False
        """
        if opt:
            tab =  pd.DataFrame(self.df.copy())
        else:
            tab = pd.DataFrame(self.dfrisk.copy())
            
        rdmt_ann = self.annualized_returns(tab, False, True)
        
        rt = [x/100 for x in pd.DataFrame(self.dfriskless.copy()).iloc[:,0]]
        rt_mean = np.mean(rt)
        
        vol = self.annualized_volatility_NAV(False)
        
        sharpe = (rdmt_ann-rt_mean)/vol
        
        if msg == True:
            if opt:
                print(f"The Sharpe Ratio of the Fund is: {round(sharpe,5)}")
            else:
                print(f"The Sharpe Ratio of the risky asset is: {round(sharpe,5)}")
        return sharpe
    
    def value_at_risk(self, alpha:float=5, msg:bool=False, opt:bool=False)->float:
        """
        Parameters:
            - alpha:float
            - msg:bool
            - opt:bool
        Returns:
            - float
        Description:
            This method computes the alpha% Value At Risk of:
                - The Fund, if opt is True,
                - The risky asset, if opt is False
        """
        if opt:
            tab = pd.DataFrame(self.df.copy())
        else:
            tab = pd.DataFrame(self.dfrisk.copy())
        table = self.pct_data(tab)
        list_return = [rtn for rtn in table.iloc[:,0]]
        var = np.percentile(list_return,alpha)
        
        if msg:
            if opt:
                print(f"The Value-At-Risk at {alpha}% of the Fund is: {var}")
            else:
                print(f"The Value-At-Risk at {alpha}% of the risky asset is: {var}")
        else:
            pass
        return var
    
    def semi_sqrt(self, msg:bool=False, opt:bool=False)->float:
        """
        Parameters:
            - msg:bool
            - opt:bool
        Returns:
            - float
        Description:
            This method computes the semi-deviation:
                - The Fund, if opt is True,
                - The risky asset, if opt is False
        """
        if opt:
            tab = pd.DataFrame(self.df.copy())
        else:
            tab = pd.DataFrame(self.dfrisk.copy())
        table = self.pct_data(tab)
        neg_returns = [rtn for rtn in table.iloc[:,0] if rtn < 0]
        nb_neg_returns = len(neg_returns)
        square_neg_returns = [neg_rtn**2 for neg_rtn in neg_returns]
        semivariance = sum(square_neg_returns)/nb_neg_returns
        semideviation = np.sqrt(semivariance)
        if msg:
            if opt:
                print(f"The semi-deviation of the Fund is: {round(semideviation*100,5)}%")
            else:
                print(f"The semi-deviation of the risky asset is: {round(semideviation*100,5)}%")
        else:
            pass
        return semideviation
    
    def expected_shortfall(self, alpha:float = 5, msg:bool=False, opt:bool=False)->float:
        """
        Parameters:
            - alpha:float
            - msg:bool
            - opt:bool
        Returns:
            - float
        Description:
            This method computes the alpha% Expected Shortfall of:
                - The Fund, if opt is True,
                - The risky asset, if opt is False
        """
        if opt:
            tab = pd.DataFrame(self.df.copy())
        else:
            tab = pd.DataFrame(self.dfrisk.copy())
        table = self.pct_data(tab)
        critical_value = self.value_at_risk(alpha, False, opt)
        list_returns = [rtn for rtn in table.iloc[:,0] if rtn < critical_value]
        ES = np.mean(list_returns)
        
        if msg:
            if opt:
                print(f"The Expected Shortfall of the Fund is: {ES}")
            else:
                print(f"The Expected Shortfall of risky asset is: {ES}")
        return ES
    
    def message(self)->None:
        """
        Description:
            This method doesn't return anything, only messages. 
        """
        self.mean(msg=True, opt=False)
        self.mean(msg=True, opt=True)
        self.annualized_volatility_NAV(True)
        self.annualized_volatility_risk(True)
        self.variation_coeff(True, True)
        self.variation_coeff(True, False)
        self.annualized_returns(self.df, msg=True, opt=True)
        self.annualized_returns(self.dfrisk, msg=True, opt=False)
        self.drawdown(self.df, msg=True,opt=True)
        self.drawdown(self.dfrisk, msg=True,opt=False)
        self.sharperatio(msg=True,opt=True)
        self.sharperatio(msg=True,opt=False)
        self.value_at_risk(alpha=5, msg=True, opt=True)
        self.value_at_risk(alpha=5, msg=True, opt=False)
        self.expected_shortfall(alpha=5, msg=True, opt=False)
        self.expected_shortfall(alpha=5, msg=True, opt=True)
        self.semi_sqrt(True, True)
        self.semi_sqrt(True, False)
        
    def summary(self)->pd.DataFrame:
        """
        Return:
            - pd.DataFrame
        Description:
            This method return a DataFrame of comparison between the Fund and the risky asset
        """
        index = ["Return's Mean (%)", "Ann. Volatility (%)", "Var. Coeff. (%)","Ann. Return (%)", "MaxDrawDown (%)", "Sharpe Ratio", "VaR 5%", 
                 "Exp. Shortfall", "Semi Deviation (%)"]
        meanNAV = self.mean(msg=False, opt=False)*100
        meanRisk = self.mean(msg=False, opt=True)*100
        volNAV = self.annualized_volatility_NAV(False)
        volRisk = self.annualized_volatility_risk(False)
        varcoeffNAV = self.variation_coeff(False, True)*100
        varcoeffRisk = self.variation_coeff(False, False)*100
        retNAV = self.annualized_returns(self.df, msg=False, opt=True)*100
        retRisk = self.annualized_returns(self.dfrisk, msg=False, opt=False)*100
        mddNAV = self.drawdown(self.df, msg=False,opt=True)*100
        mddRisk = self.drawdown(self.dfrisk, msg=False,opt=False)*100
        srNAV = self.sharperatio(msg=False,opt=True)
        srRisk = self.sharperatio(msg=False,opt=False)
        varNAV = self.value_at_risk(alpha=5, msg=False, opt=True)
        varRisk = self.value_at_risk(alpha=5, msg=False, opt=False)
        expshort_NAV = self.expected_shortfall(alpha=5, msg=False, opt=True)
        expshort_Risk = self.expected_shortfall(alpha=5, msg= False, opt=False)
        semsqrtNAV = self.semi_sqrt(False, True)*100
        semsqrtRisk = self.semi_sqrt(False, False)*100  
        NAVpart = [meanNAV, volNAV, varcoeffNAV, retNAV, mddNAV, srNAV, varNAV, expshort_NAV, semsqrtNAV]
        Riskpart = [meanRisk, volRisk, varcoeffRisk, retRisk, mddRisk, srRisk, varRisk, expshort_Risk, semsqrtRisk]
        dfNAV = pd.DataFrame({"Fund":NAVpart}, index = index)
        dfRisk = pd.DataFrame({"Risky Asset":Riskpart}, index=index)
        summary_data = pd.concat([dfNAV,dfRisk], axis=1)
        return summary_data

