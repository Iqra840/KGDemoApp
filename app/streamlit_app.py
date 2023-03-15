import streamlit as st
from streamlit.components.v1 import html as st_html
from st_aggrid import AgGrid
import numpy as np
import pandas as pd
import toml
import os
from neo4jdriver import SERVICE, Match
from typing import Union, Optional
from py2neo.matching import NodeMatcher, RelationshipMatcher


KEY = 0 # key to be entered into input fields; increment this every time
#
#_SECRETS = toml.load("streamlit/secrets.toml")
#NEODASH_URL = _SECRETS["external_links"]["neodash_url"]

NEODASH_URL="https://iqra840.github.io/nd_demo/"
nodematcher = NodeMatcher(SERVICE['fmptest'])
relationshipmatcher = RelationshipMatcher(SERVICE['fmptest'])
company_props = ['interestCoverage_T1yr', 'priceCashFlowRatio_T1yr', 'payoutRatio_T5yr', 
                 'returnOnCapitalEmployed_T5yr', 'fixedAssetTurnover_T1yr', 
                 'threeYOperatingCFGrowthPerShare_T1yr', 'pretaxProfitMargin_T1yr', 
                 'threeYShareholdersEquityGrowthPerShare_T5yr', 'payablesTurnover_T1yr', 
                 'weightedAverageSharesDilutedGrowth_T1yr', 'priceEarningsToGrowthRatio_T5yr', 
                 'tenYNetIncomeGrowthPerShare_T5yr', 'daysOfInventoryOutstanding_T1yr', 
                 'assetTurnover_T5yr', 'priceSalesRatio_T1yr', 'netIncomePerEBT_T5yr', 
                 'threeYDividendperShareGrowthPerShare_T1yr', 'freeCashFlowOperatingCashFlowRatio_T1yr', 
                 'cashFlowCoverageRatios_T5yr', 'cashConversionCycle_T5yr', 'ipoDate', 
                 'revenueGrowth_T1yr', 'receivablesTurnover_T5yr', 'quickRatio_T1yr', 'industry', 
                 'inventoryTurnover_T5yr', 'ceo', 'priceToFreeCashFlowsRatio_T1yr', 
                 'weightedAverageSharesGrowth_T5yr', 'cashRatio_T5yr', 'operatingProfitMargin_T5yr', 
                 'shortTermCoverageRatios_T5yr', 'currency', 'epsgrowth_T1yr', 
                 'tenYDividendperShareGrowthPerShare_T5yr', 'website', 'address', 
                 'freeCashFlowGrowth_T1yr', 'debtRatio_T1yr', 'currentRatio_T1yr', 
                 'grossProfitGrowth_T5yr', 'epsdilutedGrowth_T5yr', 'returnOnEquity_T1yr', 
                 'grossProfitMargin_T1yr', 'priceEarningsRatio_T1yr', 
                 'tenYRevenueGrowthPerShare_T1yr', 'ebitPerRevenue_T1yr', 
                 'totalDebtToCapitalization_T1yr', 'dividendYield_T5yr', 'symbol', 
                 'country', 'priceFairValue_T1yr', 'cusip', 'fiveYDividendperShareGrowthPerShare_T5yr', 
                 'operatingCashFlowPerShare_T5yr', 'tenYOperatingCFGrowthPerShare_T5yr', 
                 'receivablesGrowth_T5yr', 'dividendPayoutRatio_T1yr', 'cashPerShare_T1yr', 
                 'sgaexpensesGrowth_T1yr', 'debtEquityRatio_T1yr', 'debtGrowth_T1yr', 
                 'exchangeShortName', 'sector', 'priceBookValueRatio_T5yr', 
                 'image', 'inventoryGrowth_T1yr', 'netIncomeGrowth_T5yr', 
                 'returnOnAssets_T5yr', 'isFund', 'freeCashFlowPerShare_T1yr', 
                 'daysOfSalesOutstanding_T1yr', 'effectiveTaxRate_T1yr', 
                 'cashFlowToDebtRatio_T5yr', 'ebtPerEbit_T1yr', 'priceToSalesRatio_T1yr', 
                 'enterpriseValueMultiple_T1yr', 'exchange', 'capitalExpenditureCoverageRatio_T5yr', 
                 'threeYRevenueGrowthPerShare_T5yr', 'fiveYShareholdersEquityGrowthPerShare_T1yr', 
                 'operatingCashFlowSalesRatio_T5yr', 'changes', 'description', 
                 'priceToOperatingCashFlowsRatio_T1yr', 'fx', 'operatingCashFlowGrowth_T5yr', 
                 'isAdr', 'bookValueperShareGrowth_T1yr', 'ebitgrowth_T5yr', 
                 'daysOfPayablesOutstanding_T1yr', 'companyEquityMultiplier_T5yr', 
                 'threeYNetIncomeGrowthPerShare_T5yr', 'priceToBookRatio_T5yr', 
                 'dividendsperShareGrowth_T5yr', 'assetGrowth_T5yr', 
                 'fiveYRevenueGrowthPerShare_T1yr', 'netProfitMargin_T5yr', 
                 'fiveYOperatingCFGrowthPerShare_T5yr', 'operatingCycle_T5yr', 
                 'tenYShareholdersEquityGrowthPerShare_T1yr', 'operatingIncomeGrowth_T1yr', 
                 'mktCapUSD', 'dividendPaidAndCapexCoverageRatio_T1yr', 
                 'longTermDebtToCapitalization_T1yr', 'fiveYNetIncomeGrowthPerShare_T1yr', 
                 'rdexpenseGrowth_T1yr', 'priceCashFlowRatio_T5yr', 'freeCashFlowOperatingCashFlowRatio_T5yr', 
                 'companyName', 'returnOnCapitalEmployed_T1yr', 'interestCoverage_T5yr', 'payoutRatio_T1yr', 
                 'priceEarningsToGrowthRatio_T1yr', 'price', 'fixedAssetTurnover_T5yr', 'zip', 
                 'assetTurnover_T1yr', 'threeYOperatingCFGrowthPerShare_T5yr', 'pretaxProfitMargin_T5yr', 
                 'tenYNetIncomeGrowthPerShare_T1yr', 'daysOfInventoryOutstanding_T5yr', 'lastDiv', 
                 'threeYShareholdersEquityGrowthPerShare_T1yr', 'priceSalesRatio_T5yr', 
                 'payablesTurnover_T5yr', 'weightedAverageSharesDilutedGrowth_T5yr', 
                 'isActivelyTrading', 'threeYDividendperShareGrowthPerShare_T5yr', 
                 'cashFlowCoverageRatios_T1yr', 'phone', 'cashConversionCycle_T1yr', 
                 'netIncomePerEBT_T1yr', 'grossProfitMargin_T5yr', 'revenueGrowth_T5yr', 
                 'city', 'mktCap', 'priceEarningsRatio_T5yr', 'inventoryTurnover_T1yr', 
                 'range', 'debtRatio_T5yr', 'priceToFreeCashFlowsRatio_T5yr', 'currency_code', 
                 'quickRatio_T5yr', 'operatingProfitMargin_T1yr', 'volAvg', 'epsgrowth_T5yr', 
                 'weightedAverageSharesGrowth_T1yr', 'tenYDividendperShareGrowthPerShare_T1yr', 
                 'receivablesTurnover_T1yr', 'tenYRevenueGrowthPerShare_T5yr', 'ebitPerRevenue_T5yr', 
                 'shortTermCoverageRatios_T1yr', 'freeCashFlowGrowth_T5yr', 'totalDebtToCapitalization_T5yr', 
                 'returnOnEquity_T5yr', 'currentRatio_T5yr', 'epsdilutedGrowth_T1yr', 
                 'grossProfitGrowth_T1yr', 'operatingCashFlowPerShare_T1yr', 'priceFairValue_T5yr', 
                 'tenYOperatingCFGrowthPerShare_T1yr', 'receivablesGrowth_T1yr', 'dividendYield_T1yr', 
                 'fiveYDividendperShareGrowthPerShare_T1yr', 'debtEquityRatio_T5yr', 
                 'cashPerShare_T5yr', 'priceBookValueRatio_T1yr', 'debtGrowth_T5yr', 
                 'sgaexpensesGrowth_T5yr', 'defaultImage', 'inventoryGrowth_T5yr', 
                 'cashRatio_T1yr', 'netIncomeGrowth_T1yr', 'returnOnAssets_T1yr', 
                 'daysOfSalesOutstanding_T5yr', 'cashFlowToDebtRatio_T1yr', 
                 'effectiveTaxRate_T5yr', 'fullTimeEmployees', 'enterpriseValueMultiple_T5yr', 
                 'ebtPerEbit_T5yr', 'priceToSalesRatio_T5yr', 'freeCashFlowPerShare_T5yr', 
                 'capitalExpenditureCoverageRatio_T1yr', 'threeYRevenueGrowthPerShare_T1yr', 
                 'fiveYShareholdersEquityGrowthPerShare_T5yr', 'dcf', 'operatingCashFlowSalesRatio_T1yr', 
                 'operatingCashFlowGrowth_T1yr', 'assetGrowth_T1yr', 'bookValueperShareGrowth_T5yr', 
                 'ebitgrowth_T1yr', 'daysOfPayablesOutstanding_T5yr', 'beta', 
                 'threeYNetIncomeGrowthPerShare_T1yr', 'fiveYNetIncomeGrowthPerShare_T5yr', 
                 'companyEquityMultiplier_T1yr', 'fiveYRevenueGrowthPerShare_T5yr', 
                 'priceToBookRatio_T1yr', 'dividendsperShareGrowth_T1yr', 
                 'fiveYOperatingCFGrowthPerShare_T1yr', 'dividendPayoutRatio_T5yr', 
                 'isEtf', 'tenYShareholdersEquityGrowthPerShare_T5yr', 
                 'priceToOperatingCashFlowsRatio_T5yr', 'operatingIncomeGrowth_T5yr', 
                 'operatingCycle_T1yr', 'rdexpenseGrowth_T5yr', 'netProfitMargin_T1yr', 
                 'dividendPaidAndCapexCoverageRatio_T5yr', 'isin', 'longTermDebtToCapitalization_T5yr']


sidebar_options = (
    "introduction",
    "idea generator",
    # "recommender",
    "stock analyzer",
    "company earnings",
    "company segments",
    "supply chain",
    "graph explorer"
)

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == 'temporarypassword':
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():

    st.title("Impax Universe Explorer")
    st.caption("Alpha 0.0.1")
    sidebar_section = st.sidebar.radio(
        "Choose one option",
        options=sidebar_options
    )

    ############################
    ###### introduction #######
    ############################
    if sidebar_section == "introduction":
        st.header("Introduction")
        st.markdown("""
        This an AI-powered idea exploration tool. The following workflows are supported:

        ## 1. Thematic screening:
        Screen for companies by themes.
        1) Go to `idea generator`, enter a theme (keyword), and select your criteria (market cap, liquidity, return, etc).
        The system also predicts likelihoood of a company going into the A-list and its potential category.

        2) After getting a short list of ideas, go to `stock analyzer` where you can find companies similar to the stock and test themes.

        3) You can also find the most relevant strategies and analysts of said stock recommended by the system.
        
        ## 2. Strategy level screening:
        Have the system recommend stocks for you. Suitable for strategy level IG meetings.
        Go to `idea generator > recommender` and select either an analyst name or strategy name to begin

        ## 3. Company level analysis
        1) Go to `company analyzer` to test themes or find peers

        2) Go to `supply chain` to map out the exposure of a company

        3) Go to `explorer` for more cool functions

        ## 4. Earnings transcript analysis (US only at this stage)
        Find prominent keywords mentioned across company earnings calls

        ## 5. [Under development] Website / document summarizer
        Pass a document or weblink and have the system summarize them for you, and recommend related companies.

        """)



    ############################
    ######### neodash ##########
    ############################
    elif sidebar_section == "graph explorer":
        st.markdown("# Exploration")
        st_html(f"""<embed 
            type="text/html"
            height="1200"
            width="2100"
            src={NEODASH_URL}
            allowfullscreen="true" 
            style="margin-left:auto; margin-right:auto;"
            >""",
            height=1200,
            width=1800)


    ############################
    #### cluster analysis ######
    ############################

    elif sidebar_section == "stock analyzer":
        similarity, path, recommender = st.tabs(["Similar stocks", "Connection explorer", "Recommender"])
        with similarity:
            st.markdown("#Stock similarity")
            st.caption("This functions helps you find similar companies.\n Start by entering a commpany name.")
            
            company_name = st.text_input("enter your company name", key=KEY)
            KEY += 1

            clicked = st.button("check similarity")
            g = SERVICE["fmptest"]
            query = """
            CALL gds.nodeSimilarity.stream(
                    "subG1"
                    ) 
                YIELD
                    node1,
                    node2,
                    similarity
                RETURN gds.util.asNode(node1).companyName as Comp1, gds.util.asNode(node2).companyName as Comp2, similarity
                ORDER BY similarity DESCENDING
            """
            if clicked:
                res = g.run(query)
                df = res.to_data_frame().drop_duplicates()
                df = df.loc[df.Comp1.str.contains(company_name)]
                df = df.pivot_table(index="Comp1", columns="Comp2", aggfunc="mean")
                st.dataframe(df.similarity.T)
        with path:
            st.markdown("#Connection finder")
            st.caption("This tool helps you explore the potential connection between a stock and another or a theme.")


    ########################
    ### company earnings ###
    ########################
    elif sidebar_section == "company earnings":
        pass
    # TODO - after adding company earnings graph
    # TODO - 1. earnings keyword centrality
    # TODO - 2. earning keyword heatmap


    ########################
    ### idea generator ###
    ########################

    elif sidebar_section == "idea generator":
        desc_search, segment, impax_tax = st.tabs(["Search based on business description", 
                                                    "Search based on segment", 
                                                    "Search based on Impax Taxonomy"])
        with desc_search:
            # a_company = nodematcher.match("Company")
            input = st.text_input("Enter a keyword to start: ", key=KEY)
            KEY += 1


            mktcap_lb = st.number_input(label="select the lowest market cap (USD) you allow (M)",
                min_value=0., max_value=10e7, value=0., step=1., key=KEY)
            KEY += 1

            available_data_fields = company_props
            selected_data_fields = st.multiselect(label="select the fields for display",
                                                options=available_data_fields,
                                                default=["mktCapUSD", "payoutRatio_T5yr", "returnOnEquity_T5yr", 'revenueGrowth_T5yr'],
                                                key=KEY)
            KEY += 1


            def generate_query(input: str,
                            # mktcap_ub: Optional[Union[int, float]]=None,
                            mktcap_lb: Optional[Union[int, float]]=None,
                            # evebitda_up: Optional[Union[int, float]]=None,
                            # roic_lb: Optional[Union[int, float]]=None,
                            # rev_growth_lb: Optional[Union[int, float]]=None,
                            # fcst_rev_growth: Optional[Union[int, float]]=None
                            ):
                mktcap_where = f"AND n.mktCapUSD > {mktcap_lb * 1e6}"

                def get_query(thres, field_name: str, ub: bool=True, ):
                    query = f"AND n.{field_name} {'<' if ub else '>'} {thres}" if thres is not None else ""
                    return query
                # evebitda_where = get_query(evebitda_up, field_name="enterpriseValueMultipleTTM_T1yr")
                # roic_where = get_query(roic_lb, field_name="returnOnCapitalEmployedTTM_T1yr", ub=False)
                return_fields = ", ".join([f"n.{t} as {t}" for t in selected_data_fields])
                flds = ", ".join(selected_data_fields)
                query = f"""MATCH (k:Keyword) 
                WHERE k.name contains "{input}"
                WITH Collect(id(k)) as ids
                MATCH (n)-[r:mention]->(k:Keyword) 
                WHERE id(k) in ids AND r.score IS NOT NULL {mktcap_where}
                WITH n.companyName as companyName, n.symbol as Ticker, n.sector as sector, 
                (r.score * r.count) as relevanceScore,
                k.name as keywords,
                {return_fields}
                RETURN companyName, Ticker, sector, relevanceScore, keywords, {flds}
                ORDER BY relevanceScore DESC
                """
                return query

            clicked = st.button("search", key=KEY)
            KEY += 1
            if clicked:

                
                query = generate_query(input, 
                                    mktcap_lb=mktcap_lb, 
                                    #    evebitda_up=evebitda_up, 
                                    #    roic_lb=roic_lb
                                    )
                # st.write(query) # debug only
                g = SERVICE["fmptest"]
                res = g.run(query)
                df = res.to_data_frame()
                keyword_dict = {}
                sector_dict = df.set_index("companyName").sector.to_dict()
                for c in df.companyName.unique():
                    keyword_dict[c] = ", ".join(df.loc[df.companyName == c].keywords.values)
                df = df.set_index(["companyName", "Ticker", "sector"]).groupby(by="companyName").mean()
                df.insert(loc=1, column="keywords", 
                            value=df.index.get_level_values("companyName").to_series().map(keyword_dict))
                df.insert(loc=2, column="sector", 
                            value=df.index.get_level_values("companyName").to_series().map(sector_dict))
                st.write(df.sort_values(by="relevanceScore", ascending=False))
        with segment:
            st.write("Coming soon")
        with impax_tax:
            sectors = SERVICE["fmptest"].run("MATCH (n:environmentalSector) WITH n.name AS name RETURN name")
            sectors = list(sectors.to_data_frame().name.unique())

            impax_sector = st.selectbox("select a sector", options=sectors, key=KEY)
            KEY += 1


            mktcap_lb = st.number_input(label="select the lowest market cap (USD) you allow (M)",
                min_value=0., max_value=10e7, value=0., step=1., key=KEY)
            KEY += 1

            available_data_fields = company_props
            selected_data_fields = st.multiselect(label="select the fields for display",
                                                options=available_data_fields,
                                                default=["mktCapUSD", "payoutRatio_T5yr", "returnOnEquity_T5yr", 'revenueGrowth_T5yr'],
                                                key=KEY)
            KEY += 1


            def generate_query(sectorname: str,
                            # mktcap_ub: Optional[Union[int, float]]=None,
                            mktcap_lb: Optional[Union[int, float]]=None,
                            # evebitda_up: Optional[Union[int, float]]=None,
                            # roic_lb: Optional[Union[int, float]]=None,
                            # rev_growth_lb: Optional[Union[int, float]]=None,
                            # fcst_rev_growth: Optional[Union[int, float]]=None
                            ):
                mktcap_where = f"AND n.mktCapUSD > {mktcap_lb * 1e6}"

                def get_query(thres, field_name: str, ub: bool=True, ):
                    query = f"AND n.{field_name} {'<' if ub else '>'} {thres}" if thres is not None else ""
                    return query
                # evebitda_where = get_query(evebitda_up, field_name="enterpriseValueMultipleTTM_T1yr")
                # roic_where = get_query(roic_lb, field_name="returnOnCapitalEmployedTTM_T1yr", ub=False)
                return_fields = ", ".join([f"n.{t} as {t}" for t in selected_data_fields])
                flds = ", ".join(selected_data_fields)
                query = f"""MATCH (k:Keyword)<-[r]-(s:environmentalSector) 
                WHERE s.name contains "{sectorname}" AND r.score > 0.4
                WITH Collect(id(k)) as ids
                MATCH (n)-[r:mention]->(k:Keyword) 
                WHERE id(k) in ids AND r.score IS NOT NULL {mktcap_where}
                WITH n.companyName as companyName, n.symbol as Ticker, n.sector as sector, 
                (r.score * r.count) as relevanceScore,
                k.name as keywords,
                {return_fields}
                RETURN companyName, Ticker, sector, relevanceScore, keywords, {flds}
                ORDER BY relevanceScore DESC
                """
                return query

            clicked = st.button("search", key=KEY)
            KEY += 1
            if clicked:

                
                query = generate_query(impax_sector, 
                                    mktcap_lb=mktcap_lb, 
                                    #    evebitda_up=evebitda_up, 
                                    #    roic_lb=roic_lb
                                    )
                # st.write(query) # debug only
                g = SERVICE["fmptest"]
                res = g.run(query)
                df = res.to_data_frame()
                keyword_dict = {}
                sector_dict = df.set_index("companyName").sector.to_dict()
                for c in df.companyName.unique():
                    keyword_dict[c] = ", ".join(df.loc[df.companyName == c].keywords.values)
                df = df.set_index(["companyName", "Ticker", "sector"]).groupby(by="companyName").mean()
                df.insert(loc=1, column="keywords", 
                            value=df.index.get_level_values("companyName").to_series().map(keyword_dict))
                df.insert(loc=2, column="sector", 
                            value=df.index.get_level_values("companyName").to_series().map(sector_dict))
                st.write(df.sort_values(by="relevanceScore", ascending=False))


    #############################
    ### supply chain analysis ###
    #############################
    elif sidebar_section == "supply chain":
        # TODO (@Iqra) - a table for supply chain importance
        # TODO - (@Iqra) a query window for getting the entire supply chain
        pass

    ########################
    ### revenue exposure ###
    ########################

    elif sidebar_section == "revenue exposure":
        # TODO (@PP) add revenue exposure
        # TODO merge revenue exposure segments with keywords
        pass
