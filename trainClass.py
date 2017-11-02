#
from env import setEnv
import pandas as pd
import numpy as np
import argparse

from sklearn import model_selection, preprocessing

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



class trainClass(object):

    def __init__(self,test=False):

        self.envs = setEnv()
        self.test = test

        #self.readMainData()

    def readMainData(self):

        print("\n")
        print("+"*30)
        print("\n      load 2016 train data   ")
        self.train = self.readTrain2016()
        self.changeTrainYearMonthFromTransactionDate()


        print("\n")
        print("+"*30)
        print("\n      load properties data   ")
        self.prop = self.readProp2016()
        #
        #
        #
        self.setCountNonZeroFeatures()
        #
        #
        #
        print("\n")
        print("+"*30)
        print("\n      load Sample data   ")
        self.sample = self.makeSampleForTesting()

    def setCountNonZeroFeatures(self):

        print("\n")
        print("-"*30)
        print("\n    set NonZeroCounter of properties ...")
        print("-"*30)
        properties = self.getProp()
        properties["non_zero_features_counter"]  = properties.astype(bool).sum(axis=1)
        self.setProp(properties)

    def setFeaturesSumAndVar(self):

        properties=self.getProp()
        print("\n")
        print("-"*30)
        print("\n    set Sum/Var of Features of properties ...")
        print("-"*30)
        properties["features_sum"] = properties.sum(axis=1).values.reshape(-1,1)
        properties["features_var"] = properties.var(axis=1).values.reshape(-1,1)

        scaler = StandardScaler()
        target_list = ["features_sum","features_var"]
        for c in target_list:
            properties[c] = scaler.fit_transform(properties[c].values.reshape(-1,1))

        #print(properties["taxamount"].describe().transpose() )


        self.setProp(properties)

    def getSample(self):
        return self.sample

    def readProp2016(self):
        #print("\n")
        #print("-"*30)
        #print("\n      Reading properties ....")
        prop2016 = self.envs["prop"]
        return pd.read_csv(prop2016)

    def readProp2017(self):
        #print("\n")
        #print("-"*30)
        #print("\n      Reading properties ....")
        prop2017 = self.envs["prop2017"]
        return pd.read_csv(prop2017)


    def getProp(self):
        return self.prop

    def setProp(self,prop):
        self.prop = prop.copy()
        del prop

    def getTrain(self):
        return self.train

    def setTrain(self,x):
        self.train = x.copy()
        del x

    def readSampleSub(self):

        samplefile = self.envs["sample_submission"]
        df = pd.read_csv(samplefile)
        return df

    def kaggleSumbmitData(self):
        kagglefile = self.envs["orig_kaggle"]
        df  = pd.read_csv(kagglefile)
        return df

    def readTrain2016(self):
        if self.test:
            print("-"*30)
            print("\n      Reading train 2016 ....")
        train2016 = self.envs["train2016"]
        return pd.read_csv(train2016,parse_dates=["transactiondate"])

    def readTrain2017(self):
        if self.test:
            print("-"*30)
            print("\n      Reading train 2017 ....")
        train2017 = self.envs["train2017"]
        return pd.read_csv(train2017,parse_dates=["transactiondate"])


    def makeTrainingData(self,log_sw=False):
        train_df = self.getTrain()
        prop2016_df = self.getProp()

        print("     fillna with median .....")
        prop2016_df.fillna( prop2016_df.median(),inplace=True )

        data_df = train_df.merge(prop2016_df,how="left",left_on="parcelid",right_on="parcelid")
        data_df["abs_logerror"] = abs(data_df["logerror"])

        #data_df.to_csv("")
        if log_sw:
            hdf = pd.HDFStore('storage_log.h5')
        else:
            hdf = pd.HDFStore('storage_norm.h5')

        hdf.put('train', data_df, format='table', data_columns=True)
        print(hdf['train'].shape)
        #print hdf

    def makeTestingData(self,log_sw=False):

        self.addYearMonthOnProperties()
        prop2016_df = self.getProp()
        print("     fillna with median .....")
        prop2016_df.fillna( prop2016_df.median(),inplace=True )
        #if log_sw:
        #    hdf = pd.HDFStore('properties_log.h5')
        #else:
        hdf = pd.HDFStore('properties_norm.h5')
        hdf.put('properties', prop2016_df, format='table', data_columns=True)
        #print(hdf['properties'].shape)
        #print hdf
        #prop2016_df.to_csv("/Users/donchan/Documents/myData/KaggleData/zillow/prop_custom.csv",index=False)

    def addFeatures(self,_df):

        df_train = _df.copy()

        print("-"*30)
        print("\n    addtional features LivingArea ValueRatio etc .....   ")

        #proportion of living area
        df_train['N-LivingAreaProp'] = df_train['calculatedfinishedsquarefeet']/df_train['lotsizesquarefeet']
        #Ratio of tax of property over parcel
        df_train['N-ValueRatio'] = df_train['taxvaluedollarcnt']/df_train['taxamount']
        #Ratio of the built structure value to land area
        df_train['N-ValueProp'] = df_train['structuretaxvaluedollarcnt']/df_train['landtaxvaluedollarcnt']
        # latitude + longitue
        df_train["N-location"] = df_train["latitude"] + df_train["longitude"]


        #Average structuretaxvaluedollarcnt by city
        group = df_train.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
        df_train['N-Avg-structuretaxvaluedollarcnt'] = df_train['regionidcity'].map(group)

        #Deviation away from average
        df_train['N-Dev-structuretaxvaluedollarcnt'] = abs((df_train['structuretaxvaluedollarcnt'] -
                                                                df_train['N-Avg-structuretaxvaluedollarcnt']))/df_train['N-Avg-structuretaxvaluedollarcnt']

        return df_train

    def ratioCalc(self):

        prop = self.getProp()

        print("-"*30)
        print("\n    properties calc ratio .....   ")

        #'transaction_year', 'transaction_month', 'transaction_qtr',
        #'non_zero_features_counter', 'latitude', 'region_county',
        #'zoning_landuse', 'longitude', 'fips', 'tax_year', 'num_bedroom',
        #'num_bathroom', 'num_room', 'zoning_landuse_county', 'region_zip',
        #'tax_property', 'tax_total', 'tax_building', 'area_total_calc',
        #'build_year', 'region_city', 'tax_land', 'censustractandblock',
        #'num_bathroom_calc', 'num_bath', 'area_live_finished', 'area_lot',
        #'tract', 'block', 'build_category_Latest_build',
        #'build_category_Old_build', 'build_category_VeryOld_build',
        #'features_sum', 'features_var', 'over_under_range_latitude',
        #'over_under_range_longitude']

        prop = prop.assign(livingAreaProp = lambda x:x.area_total_calc / x.area_lot  )
        prop = prop.assign(N_ValueRatio = lambda x:x.tax_total / x.tax_property  )
        prop = prop.assign(N_ValueProp = lambda x:x.tax_building / x.tax_land  )

        group = prop.groupby("region_city")["tax_building"].aggregate('mean').to_dict()
        prop["N_Avg_structuretaxvaluedollarcnt"] = prop["region_city"].map(group)


        prop = prop.assign(tax_property_bath_ratio = lambda x:x.tax_property / (x.num_bathroom + 1)  )
        prop = prop.assign(tax_total_bath_ratio = lambda x:x.tax_total / (x.num_bathroom + 1)  )
        prop = prop.assign(tax_building_bath_ratio = lambda x:x.tax_building / (x.num_bathroom + 1)  )
        prop = prop.assign(tax_land_bath_ratio = lambda x:x.tax_land / (x.num_bathroom + 1)  )

        prop = prop.assign(tax_land_room_ratio = lambda x:x.tax_land / (x.num_room + 1)  )
        prop = prop.assign(tax_total_room_ratio = lambda x:x.tax_total / (x.num_room + 1)  )
        prop = prop.assign(tax_building_room_ratio = lambda x:x.tax_building / (x.num_room + 1)  )
        prop = prop.assign(tax_land_room_ratio = lambda x:x.tax_land / (x.num_room + 1)  )

        prop = prop.assign(tax_land_bedroom_ratio = lambda x:x.tax_land / (x.num_bedroom + 1)  )
        prop = prop.assign(tax_total_bedroom_ratio = lambda x:x.tax_total / (x.num_bedroom + 1)  )
        prop = prop.assign(tax_building_bedroom_ratio = lambda x:x.tax_building / (x.num_bedroom + 1)  )
        prop = prop.assign(tax_land_bedroom_ratio = lambda x:x.tax_land / (x.num_bedroom + 1)  )

        prop = prop.assign(area_total_calc_room_ratio = lambda x:x.area_total_calc / (x.num_room + 1)  )
        prop = prop.assign(area_live_finished_room_ratio = lambda x:x.area_live_finished / (x.num_room + 1)  )
        prop = prop.assign(area_lot_room_ratio = lambda x:x.area_lot / (x.num_room + 1)  )


        self.setProp(prop)

    def addFeatures_year(self,_prop):

        ############
        #    additional fields for testing ...
        #
        prop = _prop.copy()

        print("-"*30)
        print("\n    addtional features yearbuilt bedroomcnt etc .....   ")

        #########################################################################################3
        ####Ofert1: LIGTH to 0.0652813
        ofert1 = prop.groupby(['yearbuilt', 'bedroomcnt', 'regionidcity'],  as_index=False)['parcelid'].count()
        ofert1=pd.DataFrame(ofert1)
        ofert1.columns.values[3] = 'count_ParcelId'
        prop= pd.merge(prop,ofert1, on=['yearbuilt', 'bedroomcnt', 'regionidcity'], how='left')



        ####Ofert2: v12
        ofert2 = prop.groupby(['yearbuilt', 'roomcnt', 'regionidcity'],  as_index=False)['parcelid'].count()
        ofert2=pd.DataFrame(ofert2)
        ofert2.columns.values[3] = 'count_ParcelId_Of2'
        prop= pd.merge(prop,ofert2, on=['yearbuilt', 'roomcnt', 'regionidcity'], how='left')


        ####Ofert3:v12
        ofert3 = prop.groupby(['yearbuilt', 'bathroomcnt', 'regionidcity'],  as_index=False)['parcelid'].count()
        ofert3=pd.DataFrame(ofert3)
        ofert3.columns.values[3] = 'count_ParcelId_Of3'
        prop= pd.merge(prop,ofert3, on=['yearbuilt', 'bathroomcnt', 'regionidcity'], how='left')


        ####Ofert4: v12
        ofert4 = prop.groupby(['yearbuilt', 'finishedsquarefeet12', 'regionidcity'],  as_index=False)['parcelid'].count()
        ofert4=pd.DataFrame(ofert4)
        ofert4.columns.values[3] = 'count_ParcelId_Of4'
        prop= pd.merge(prop,ofert4, on=['yearbuilt', 'finishedsquarefeet12', 'regionidcity'], how='left')




        ####Tax1: ####Ofert1: LIGTH to 0.0652813

        Tax1 = prop.groupby(['yearbuilt', 'bedroomcnt', 'regionidcity'],  as_index=False)['taxamount'].mean()
        Tax1=pd.DataFrame(Tax1)
        Tax1.columns.values[3] = 'mean_TaxAmount'
        prop= pd.merge(prop,Tax1, on=['yearbuilt', 'bedroomcnt', 'regionidcity'], how='left')

        return prop

    def trainMaskOutlier(self):
        mask_l = train_df.logerror > -0.4
        mask_u = train_df[mask].logerror < 0.419
        return mask_u

    def setLog(self):

        # big number should be converted to log .....
        properties = self.getProp()
        for c in properties.columns.tolist():
            if "sqft" in c or "feet" in c or "block" in c:
                print("\ncolumns --> %s " % c)
                print("\n   converted to log .... ")
                properties[c] = properties[c].fillna(properties[c].min())
                #print(properties[c].min(),properties[c].max())
                properties[c] = properties[c].apply(lambda x:np.log( abs(x)+1e-7))

        self.setProp(properties)

    def setNormalize(self):

        # big number should be converted to log .....
        properties = self.getProp()
        scaler = StandardScaler()
        for c in properties.columns.tolist():
            if "sqft" in c or "feet" in c or "block" in c:
                print("\ncolumns --> %s " % c)
                print("\n   normalized with StandardScaler .... ")
                properties[c].fillna(0.0, inplace=True)
                #print(properties[c].min(),properties[c].max())
                properties[c] = scaler.fit_transform(properties[c].values.reshape(-1,1))
                print(properties[c].values[:10])




        self.setProp(properties)


    def CountFillWithMedian(self):
        #
        #
        # -----------field name ----
        # xxxxCnt
        # xxxxyear
        # --------------------
        #
        #
        # count --> 1. set with median (Aug.20th)
        #
        #
        properties = self.getProp()
        for c in properties.columns.tolist():
            if "cnt" in c or "year" in c or "nbr" in c:
                print("\ncolumns :%s be filled with median..." % c)
                mask = properties[c].isnull()
                properties.loc[mask,c] = properties[c].median()


        print("\ncolumns --> %s " % "taxamount")
        print("\n   normalized with StandardScaler .... ")
        scaler = StandardScaler()
        properties["taxamount"].fillna(0.0, inplace=True)
        #print(properties[c].min(),properties[c].max())
        properties["taxamount"] = scaler.fit_transform(properties["taxamount"].values.reshape(-1,1))
        print(properties["taxamount"].values[:10])



        self.setProp(properties)

    def makeSampleForTesting(self):

        return self.readSampleSub()

    def set_feature1(self):
        properties = self.getProp()

        #droplist = ["propertyzoningdesc","propertycountylandusecode","fireplaceflag"]
        #properties.drop(droplist, axis=1, inplace=True)

        # use LabelEncoder for object
        for c in properties.columns:
            #properties[c]=properties[c].fillna(-1)
            if properties[c].dtype == 'object':
                print("\nColumns ..... %s" % c)
                print(" change to float and fill with 0..")
                properties[c].fillna(0, inplace=True)
                lbl = LabelEncoder()
                lbl.fit(list(properties[c].values))
                properties[c] = lbl.transform(list(properties[c].values))
                properties[c] = properties[c].astype(np.float64)

        self.setProp(properties)
        #return properties.astype(float)
    def set_ohe1(self):

        #
        #
        # create one hot coder
        #
        # first, object data --> propertycountylandusecode, taxdelinquencyflag

        # second --> fips
        #
        ohe_features_list = ["airconditioningtypeid","architecturalstyletypeid",
            "buildingclasstypeid","buildingqualitytypeid","decktypeid","heatingorsystemtypeid",
            "pooltypeid10","pooltypeid2","pooltypeid7","propertylandusetypeid",
            "regionidcounty","storytypeid","typeconstructiontypeid","fips","numberofstories"]

        properties = self.getProp()
        droplist = ["propertyzoningdesc","fireplaceflag","propertycountylandusecode"]
        properties.drop(droplist, axis=1, inplace=True)
        # use LabelEncoder for object
        for c in properties.columns:
            #properties[c]=properties[c].fillna(-1)
            if properties[c].dtype == 'object':
                #if c == "propertyzoningdesc":
                #    continue
                print("\nColumns ..... %s" % c)
                print(" change to float and fill with 0..")
                properties = pd.concat([properties, pd.get_dummies(properties[c],
                                    prefix=c)], axis=1)
                properties.drop([c], axis=1, inplace=True)
            else:
                if c in ohe_features_list:
                    print("\nColumns ..... %s" % c)
                    print(" one hot code changed.....")
                    properties[c].fillna(-9999,inplace=True)
                    properties[c] = properties[c].astype(np.int32)
                    #properties[c] = pd.to_numeric(properties[c],downcast="integer",errors="coerce")
                    properties = pd.concat([properties, pd.get_dummies(properties[c],
                                        prefix=c)], axis=1)
                    c_9999 = c + "_-9999"
                    properties.drop([c,c_9999], axis=1, inplace=True)


        self.setProp(properties)

    def changeTrainYearMonthFromTransactionDate(self):

        df = self.getTrain()

        print("\n")
        print("-"*40)
        print("\n      Changing train transactiondate --> Year / Month")
        df = df.assign( transactiondate = lambda x: pd.to_datetime(x.transactiondate)   )
        df = df.assign( transaction_year = lambda x: x.transactiondate.dt.year   )
        df = df.assign( transaction_month = lambda x: x.transactiondate.dt.month   )
        df = df.assign( transaction_qtr = lambda x: x.transactiondate.dt.quarter   )
        #df = df.assign( transaction_qtrstr = lambda x: x.transactiondate.dt.to_period("Q")   )
        df = df.drop(['transactiondate'], axis=1)
        self.setTrain(df)

    def addYearMonthOnProperties(self):
        properties = self.getProp()

        print("\n")
        print("-"*40)
        print("\n     transactiondate on properties --> Year / Month")
        properties = properties.assign( transactiondate = "2016-10-01")
        properties = properties.assign( transactiondate = lambda x: pd.to_datetime(x.transactiondate)   )
        properties = properties.assign( transaction_year = lambda x: x.transactiondate.dt.year   )
        properties = properties.assign( transaction_month = lambda x: x.transactiondate.dt.month   )
        properties = properties.assign( transaction_qtr = lambda x: x.transactiondate.dt.quarter   )
        #properties = properties.assign( transaction_qtrstr = lambda x: x.transactiondate.dt.to_period("Q")   )
        properties = properties.drop(['transactiondate'], axis=1)
        self.setProp(properties)

    def changeGeoData(self):
        properties = self.getProp()

        print("\n")
        print("-"*40)
        print("\n      geo data divide 1e6")

        properties["latitude"] = properties["latitude"].apply(lambda x:x / 1e6)
        properties["longitude"] = properties["longitude"].apply(lambda x:x / 1e6)

        properties["over_under_range_latitude"] = properties["latitude"].apply(lambda x:1 if x >= 33.8 and x <= 34.15 else 0 )
        properties["over_under_range_longitude"] = properties["longitude"].apply(lambda x:1 if x >= -118.5 and x <= -118.25 else 0 )

        self.setProp(properties)


    def changeColNames(self):

        #train_df = self.getTrain()
        #train_df = train_df.rename( columns = {'parcelid':'id_parcel'})
        #self.setTrain(train_df)

        prop2016_df = self.getProp()
        prop_df = prop2016_df.rename(
            columns = {
                "yearbuilt":"build_year",
                "basementsqft":"area_basement",
                "yardbuildingsqft17":"area_patio",
                "yardbuildingsqft26":"area_shed",
                "poolsizesum":"area_pool",
                "lotsizesquarefeet":"area_lot",
                "garagetotalsqft":"area_garage",
                "finishedfloor1squarefeet":"area_firstfloor_finished",
                "calculatedfinishedsquarefeet":"area_total_calc",
                "finishedsquarefeet6":"area_base",
                "finishedsquarefeet12":"area_live_finished",
                "finishedsquarefeet13":"area_liveperi_finished",
                "finishedsquarefeet15":"area_total_finished",
                "finishedsquarefeet50":"area_unknown",
                "unitcnt":"num_unit",
                "numberofstories":"num_story",
                "roomcnt":"num_room",
                "bathroomcnt":"num_bathroom",
                "bedroomcnt":"num_bedroom",
                "calculatedbathnbr":"num_bathroom_calc",
                "fullbathcnt":"num_bath",
                "threequarterbathnbr":"num_75_bath",
                "fireplacecnt":"num_fireplace",
                "poolcnt":"num_pool",
                "garagecarcnt":"num_garage",
                "regionidcounty":"region_county",
                "regionidcity":"region_city",
                "regionidzip":"region_zip",
                "regionidneighborhood":"region_neighbor",
                "taxvaluedollarcnt":"tax_total",
                "structuretaxvaluedollarcnt":"tax_building",
                "landtaxvaluedollarcnt":"tax_land",
                "taxamount":"tax_property",
                "assessmentyear":"tax_year",
                "taxdelinquencyflag":"tax_delinquency",
                "taxdelinquencyyear":"tax_delinquency_year",
                "propertyzoningdesc":"zoning_property",
                "propertylandusetypeid":"zoning_landuse",
                "propertycountylandusecode":"zoning_landuse_county",
                "fireplaceflag":"flag_fireplace",
                "hashottuborspa":"flag_tub",
                "buildingqualitytypeid":"quality",
                "buildingclasstypeid":"framing",
                "typeconstructiontypeid":"material",
                "decktypeid":"deck",
                "storytypeid":"story",
                "heatingorsystemtypeid":"heating",
                "airconditioningtypeid":"aircon",
                "architecturalstyletypeid":"architectural_style"
             }
        )

        print("")
        print("-"*40)
        print("     Check Columns names after changing to realistic ones .......")
        print(prop_df.columns.values)
        self.setProp(prop_df)

    def split_rawcensus(self):

        properties = self.getProp()

        print("")
        print("-"*40)
        print("    rawcensus : fips / tract / block splittd ...")
        print("")

        properties["rawcensustractandblock"].fillna(properties["rawcensustractandblock"].median(), inplace=True )
        properties["tract"] = properties["rawcensustractandblock"].apply(lambda x:pd.to_numeric( str(x)[4:11] ) )
        properties["block"] = properties["rawcensustractandblock"].apply(lambda x:pd.to_numeric( str(x)[-1] ) )
        print("-"*40)
        print("     columns name after splitting rawcensustractandblock ....")

        properties.drop(["rawcensustractandblock"],axis=1, inplace=True)

        print(properties.columns.values)

        self.setProp(properties)

    def split_buildyear(self):

        properties = self.getProp()
        properties["build_year"].fillna(properties["build_year"].mean() ,inplace=True)

        properties["build_category"] = properties["build_year"].apply(lambda x:"VeryOld_build" if x < 1940 else ("Old_build" if x < 1980 else "Latest_build") )

        c = "build_category"
        print("\nColumns ..... %s" % c)
        print(" one hot code changed.....")

        #properties[c] = properties[c].astype(np.int32)
        properties = pd.concat([properties, pd.get_dummies(properties[c],
                            prefix=c)], axis=1)
        #c_9999 = c + "_-9999"
        properties.drop([c], axis=1, inplace=True)

        self.setProp(properties)

    def checkGoodColumns(self):

        prop2016_df = self.getProp()

        null_data = prop2016_df.isnull().sum()
        null_df = pd.DataFrame(data=null_data,columns=["cnt"])
        null_df["colname"] = null_df.index
        null_df.reset_index(drop=True,inplace=True)
        null_df = null_df[["colname","cnt"]].sort_values("cnt",ascending=True)
        print(null_df)

        top_columns = null_df[null_df.cnt < prop2016_df.shape[0] * .25].colname.tolist()
        prop2016_df = prop2016_df[top_columns]
        print("")
        print("-"*30)
        print("     Only less missing columns selected from prop...")
        print(prop2016_df.columns.values)
        self.setProp(prop2016_df)

    def checkMissingColumns(self):
        properties = self.getProp()
        missing_col = properties.columns[properties.isnull().any()].tolist()
        print("\n")
        print("-"*40)
        print(missing_col)
        #for col in missing_col:
        #    print("%s fill with zero ....",col)
        #    properties[col].fillna( 0,inplace=True )
        print('\n** There are {} missing columns'.format(len(missing_col)))

        for c in missing_col:
            print("\n   %s be filled with float ZERO...  " % c)
            properties[c] = properties[c].fillna(0.0)


        #print(properties["taxamount"].describe().transpose() )

        self.setProp(properties)

    def checkDataTypes(self):
        properties = self.getProp()
        print("\n")
        print("-"*40)
        print( "\n** checking data types for properties data ...")
        for col in properties.columns:
            if properties[col].dtype != object:
                print("%s %s" % (col, properties[col].dtype))
            else:
                print("%s %s : OBJECT !!  " % (col, properties[col].dtype))

        #print(properties["taxamount"].describe().transpose() )

        train = self.getTrain()
        print("\n")
        print("-"*40)
        print( "\n** checking data types for train data ...")
        for col in train.columns:
            if train[col].dtype != object:
                print("%s %s" % (col, train[col].dtype))
            else:
                print("%s %s : OBJECT !!  " % (col, train[col].dtype))


        #self.setProp(properties)

    def changeDataTypes(self):
        #
        # change data type from float64 to float32
        #
        properties = self.getProp()
        print("\n")
        print("-"*40)
        print( "\n** change data types float64 --> float32 ...")
        for col in properties.columns:
            if properties[col].dtype != object:
                if properties[col].dtype == float:
                    properties[col] = properties[col].astype(np.float32)

        #print(properties["taxamount"].describe().transpose() )

                    #print("%s %s" % (col, properties[col].dtype))
        self.setProp(properties)

    def fillRegionId(self):
        properties = self.getProp()

        #
        # replace with Random value
        #
        #  regionidcity
        #
        print("\n")
        print("-"*40)
        print( "\n** replace regionidxxxx with random value ...")

        min_ = properties["regionidcity"].min()
        max_ = properties["regionidcity"].max()
        mask = properties["regionidcity"].isnull()
        dfrand = pd.DataFrame(data = np.random.randint(min_,max_, (len(properties["regionidcity"].values )  ) ))
        properties.loc[mask, "regionidcity"] = dfrand[mask].values

        #
        #  regionidneighborhood
        #
        min_ = properties["regionidneighborhood"].min()
        max_ = properties["regionidneighborhood"].max()
        mask = properties["regionidneighborhood"].isnull()
        dfrand = pd.DataFrame(data = np.random.randint(min_,max_, (len(properties["regionidneighborhood"].values )  ) ))
        properties.loc[mask, "regionidneighborhood"] = dfrand[mask].values

        #
        #  regionidzip
        #
        min_ = properties["regionidzip"].min()
        max_ = properties["regionidzip"].max()
        mask = properties["regionidzip"].isnull()
        dfrand = pd.DataFrame(data = np.random.randint(min_,max_, (len(properties["regionidzip"].values )  ) ))
        properties.loc[mask, "regionidzip"] = dfrand[mask].values

        self.setProp(properties)

    def fillWith2016_yearbuilt(self):
        properties = self.getProp()
        print("\n")
        print("-"*30)
        print("\n      fillna yearbuilt with 2016....")
        properties['yearbuilt'].fillna(2016,inplace=True)
        self.setProp(properties)

    def convertLog(self):

        logConvertColumns = ["sqft","feet"]


    def ohe_feature_selection(self):

        ghsc1 = ["finishedsquarefeet12","taxamount","taxvaluedollarcnt"]
        #ghsc2 = ["calculatedfinishedsquarefeet","regionidzip","structuretaxvaluedollarcnt"]
        #ghsc3 = ["yearbuilt","latitude","landtaxvaluedollarcnt","longitude"]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logconv", help="logconvert",
                        action="store_true")
    args = parser.parse_args()

    log_sw = False
    if args.logconv:
        print("\n")
        print("    numbers are converted as log ..  ")
        log_sw = True
    else:
        print("\n")
        print("    numbers are converted to normalized ..  ")



    myProc2(log_sw)


def myProc2(log_sw=False):

    trainCls = trainClass()
    trainCls.readMainData()

    trainCls.changeColNames()
    trainCls.checkGoodColumns()
    trainCls.split_rawcensus()
    trainCls.split_buildyear()

    trainCls.ratioCalc()
    trainCls.set_feature1()
    trainCls.setFeaturesSumAndVar()

    trainCls.changeGeoData()
    trainCls.checkDataTypes()


    #
    # write train and properties data (customized ..)
    #
    #trainCls.changeDataTypes()

    trainCls.makeTrainingData(log_sw)
    trainCls.makeTestingData(log_sw)


def myProc(log_sw=False):

    trainCls = trainClass()
    trainCls.readMainData()

    trainCls.changeDataTypes()


    trainCls.set_ohe1()
    if log_sw:
        trainCls.setLog()
    else:
        trainCls.setNormalize()

    #trainCls.set_feature1()
    # data fill section
    trainCls.fillRegionId()
    #trainCls.addYearMonth()
    trainCls.fillWith2016_yearbuilt()
    trainCls.changeGeoData()
    trainCls.CountFillWithMedian()

    trainCls.setFeaturesSumAndVar()
    trainCls.checkMissingColumns()
    trainCls.checkDataTypes()

if __name__ == "__main__":
    main()
