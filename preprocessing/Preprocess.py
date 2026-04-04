from CONSTANTS import *
from entities.instances import Instance
from preprocessing.dataloader.BGLLoader import BGLLoader
from preprocessing.dataloader.HDFSLoader import HDFSLoader
import pandas as pd



class PKLPreprocessor:
    def __init__(self):
        self.train_event2idx = {}
        self.test_event2idx = {}
        self.id2label = {}
        self.label2id = {}
        self.embedding = None
        self.logger = None  # Optional: add logging if needed

    def load_pkl(self, dataset, train_pkl, dev_pkl=None, test_pkl=None):
        """Load train/dev/test datasets from PKL files and convert to Instance objects."""
        train_df = pd.read_pickle(train_pkl)
        dev_df = pd.read_pickle(dev_pkl) if dev_pkl else None
        test_df = pd.read_pickle(test_pkl) if test_pkl else None
        train_df.info()
        dev_df.info()
        test_df.info()
        if dataset=='TH_1G' :
            train_df = train_df.withColumn("Label", when(col("Label") == "-", "Normal").when(col("Label").isin("Normal", "Anomaly"),
                                                                                 col("Label")).otherwise("Anomaly"))
            dev_df = dev_df.withColumn("Label",
                                       when(col("Label") == "-", "Normal").when(col("Label").isin("Normal", "Anomaly"),
                                                                                col("Label")).otherwise("Anomaly"))
            test_df = test_df.withColumn("Label",
                                   when(col("Label") == "-", "Normal").when(col("Label").isin("Normal", "Anomaly"),
                                                                            col("Label")).otherwise("Anomaly"))


            print('-----check results -----------------------------------------------------')

            print('-----Training------')
            print(train_df["Original_Label"].unique())
            print(train_df["Label"].unique())

            print('-----dev ------')
            print(dev_df["Original_Label"].unique())
            print(dev_df["Label"].unique())

            print('-----test ------')
            print(test_df["Original_Label"].unique())
            print(test_df["Label"].unique())
        else:

            print('-----check results -----------------------------------------------------')

            print('-----Training------')
            print(train_df["Original_Label"].unique())
            print(train_df["Label"].unique())

            print('-----dev ------')
            print(dev_df["Original_Label"].unique())
            print(dev_df["Label"].unique())

            print('-----test ------')
            print(test_df["Original_Label"].unique())
            print(test_df["Label"].unique())



        #exit()


        train_df = train_df.drop(columns=['Label'])
        test_df = test_df.drop(columns=['Label'])
        dev_df = dev_df.drop(columns=['Label'])

        train_df = train_df.rename(columns={'Original_Label': 'Label'})
        test_df = test_df.rename(columns={'Original_Label': 'Label'})
        dev_df = dev_df.rename(columns={'Original_Label': 'Label'})

        def fix_labels(df):
            for idx, row in df.iterrows():
                current_label = row['Label']

                if current_label == 'Normal':
                    df.at[idx, 'Label'] = 'Normal'

                elif current_label == 'Anomaly':
                    df.at[idx, 'Label'] = 'Anomaly'

                else:
                    if current_label == '-':
                        df.at[idx, 'Label'] = 'Normal'
                    else:
                        df.at[idx, 'Label'] = 'Anomaly'

            return df

        train_df = fix_labels(train_df)
        dev_df = fix_labels(dev_df)
        test_df = fix_labels(test_df)


        train_df.info()
        test_df.info()
        dev_df.info()

        print(train_df['Label'].unique())
        print(test_df['Label'].unique())
        print(dev_df['Label'].unique())

        print("Train labels:", train_df['Label'].unique())
        if dev_df is not None:
            print("Dev labels:", dev_df['Label'].unique())
        if test_df is not None:
            print("Test labels:", test_df['Label'].unique())

        print(dataset)
        #exit()

        train = self._df_to_instances(train_df)
        dev = self._df_to_instances(dev_df) if dev_df is not None else []
        test = self._df_to_instances(test_df) if test_df is not None else []

        return train, dev, test

    def _df_to_instances(self, df):
        """
        Convert a DataFrame into a list of Instance objects.
        Groups by Node_block_id, preserves EventId sequence.
        """
        instances = []

        if df is None or df.empty:
            return instances

        # Make sure EventId is treated as string (do not convert to int if not numeric)
        df['EventId'] = df['EventId'].astype(str)

        grouped = df.groupby('Node_block_id')
        for block_id, g in grouped:
            # Event sequence
            sequence = g['EventId'].tolist()
            # Label: if any row is 'Anomaly', the whole block is 'Anomaly'
            label = 'Normal'
            if (g['Label'] == 'Anomaly').any():
                label = 'Anomaly'

            # Create Instance using correct argument names
            inst = Instance(block_id=block_id, log_sequence=sequence, label=label)
            instances.append(inst)

        return instances

    def _build_embedding(self, instances):
        events = set()
        for inst in instances:
            events.update(inst.sequence)
        events = sorted(events)
        self.embedding = {e: idx for idx, e in enumerate(events)}

    def _update_event2idx(self, train, test):
        for inst in train:
            for e in inst.sequence:
                if e not in self.train_event2idx:
                    self.train_event2idx[e] = len(self.train_event2idx)

        for inst in test:
            for e in inst.sequence:
                if e in self.train_event2idx:
                    self.test_event2idx[e] = self.train_event2idx[e]
                else:
                    self.test_event2idx[e] = len(self.train_event2idx)
