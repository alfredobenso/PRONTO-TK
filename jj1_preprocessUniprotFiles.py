import os
import torch
import wget
import time
from transformers import T5EncoderModel, T5Tokenizer
import h5py
import requests
import pandas as pd

def downloadReqs(DLdataFolder, url):
    filename = wget.download(url, out=DLdataFolder)
    return filename

def downloadReqs2(DLdataFolder,url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(DLdataFolder+local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def df_2_fasta(df, idName, seqName, fileOutput, maxIndex = False):
    for index, row in df.iterrows():
        if maxIndex and index>maxIndex:
            break
        uniprot_id = row[idName]
        sequence = row[seqName]
        header = f">{uniprot_id}\n{sequence}\n"
        fileOutput.write(header)

def df_2_label(df, idName, scoreName, otherName, fileOutput, maxIndex = False):
    for index, row in df.iterrows():
        if maxIndex and index>maxIndex:
            break
        header = f"{row[idName]},{row[scoreName]},{row[otherName]}\n"
        fileOutput.write(header)

def computeEmbeddings(cfg, logger):
    class ConvNet(torch.nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            # This is only called "elmo_feature_extractor" for historic reason
            # CNN weights are trained on ProtT5 embeddings
            self.elmo_feature_extractor = torch.nn.Sequential(
                torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
                torch.nn.ReLU(),
                torch.nn.Dropout(0.25),
            )
            n_final_in = 32
            self.dssp3_classifier = torch.nn.Sequential(
                torch.nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), padding=(3, 0))  # 7
            )

            self.dssp8_classifier = torch.nn.Sequential(
                torch.nn.Conv2d(n_final_in, 8, kernel_size=(7, 1), padding=(3, 0))
            )
            self.diso_classifier = torch.nn.Sequential(
                torch.nn.Conv2d(n_final_in, 2, kernel_size=(7, 1), padding=(3, 0))
            )

        def forward(self, x):
            # IN: X = (B x L x F); OUT: (B x F x L, 1)
            x = x.permute(0, 2, 1).unsqueeze(dim=-1)
            x = self.elmo_feature_extractor(x)  # OUT: (B x 32 x L x 1)
            d3_Yhat = self.dssp3_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
            d8_Yhat = self.dssp8_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 8)
            diso_Yhat = self.diso_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 2)
            return d3_Yhat, d8_Yhat, diso_Yhat

    def load_sec_struct_model(device):
        checkpoint_dir = os.path.join(cfg["ENVIRONMENT"]["t5secstructfolder"],"secstruct_checkpoint.pt")

        if not(os.path.exists(checkpoint_dir)):
            downloadReqs(cfg["ENVIRONMENT"]["t5secstructfolder"],
                     'http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt')

        #state = torch.load(checkpoint_dir).to(device)#, dtype=torch.float32)
        state = torch.load(checkpoint_dir, map_location='cpu')#, dtype=torch.float32)

        #model = torch.load(model_dir, map_location='cpu')
        #net = newModel1().float().to(device)

        model = ConvNet()
        model.load_state_dict(state['state_dict'])
        model = model.eval()
        model = model.to(device)
        logger.log_message('Loaded sec. struct. model from epoch: {:.1f}'.format(state['epoch']))
        return model

    # @title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
    # Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)

    def get_T5_model(cfg, device):
        model = T5EncoderModel.from_pretrained("Rostlab/" + cfg["EMBEDDINGS"]["model"]) # prot_t5_xl_half_uniref50-enc")  # prot_t5_xl_half_uniref50-enc prot_t5_xl_uniref50
        model = model.to(device)  # move model to GPU
        model = model.eval()  # set model to evaluation model
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        return model, tokenizer

    # @title Generate embeddings. { display-mode: "form" }
    # Generate embeddings via batch-processing
    # per_residue indicates that embeddings for each residue in a protein should be returned.
    # per_protein indicates that embeddings for a whole protein should be returned (average-pooling)
    # max_residues gives the upper limit of residues within one batch
    # max_seq_len gives the upper sequences length for applying batch-processing
    # max_batch gives the upper number of sequences per batch
    def get_embeddings(model, tokenizer, seqs, per_residue, per_protein, sec_struct,
                       max_residues=4000, max_seq_len=1000, max_batch=100):
        if sec_struct:
            sec_struct_model = load_sec_struct_model(device)

        results = {"residue_embs": dict(),
                   "protein_embs": dict(),
                   "sec_structs": dict()
                   }

        # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
        seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
        start = time.time()
        batch = list()

        #I want to set the terminator of the logger to "" to avoid the newline at the end of the log message
        for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
            seq = seq
            seq_len = len(seq)
            seq = ' '.join(list(seq))
            batch.append((pdb_id, seq, seq_len))

            # count residues in current batch and add the last sequence length to
            # avoid that batches with (n_res_batch > max_residues) get processed
            n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
            if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(
                    seq_dict) or seq_len > max_seq_len:
                pdb_ids, seqs, seq_lens = zip(*batch)
                batch = list()

                # add_special_tokens adds extra token at the end of each sequence
                token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
                input_ids = torch.tensor(token_encoding['input_ids']).to(device)
                attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

                try:
                    with torch.no_grad():
                        # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                        embedding_repr = model(input_ids, attention_mask=attention_mask)
                        #logger.log_message(".")
                except RuntimeError:
                    logger.log_message("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                    continue

                if sec_struct:  # in case you want to predict secondary structure from embeddings
                    d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)

                for batch_idx, identifier in enumerate(pdb_ids):  # for each protein in the current mini-batch
                    s_len = seq_lens[batch_idx]
                    # slice off padding --> batch-size x seq_len x embedding_dim
                    emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                    if sec_struct:  # get classification results
                        results["sec_structs"][identifier] = torch.max(d3_Yhat[batch_idx, :s_len], dim=1)[
                            1].detach().cpu().numpy().squeeze()
                    if per_residue:  # store per-residue embeddings (Lx1024)
                        results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                    if per_protein:  # apply average-pooling to derive per-protein embeddings (1024-d)
                        protein_emb = emb.mean(dim=0)
                        results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

        passed_time = time.time() - start
        if (per_residue or per_protein):
            avg_time = passed_time / len(results["residue_embs"]) if per_residue else passed_time / len(
            results["protein_embs"])
        else:
            avg_time = 0
        logger.log_message('############# EMBEDDING STATS #############')
        logger.log_message('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
        logger.log_message('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
        logger.log_message("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
            passed_time / 60, avg_time))
        logger.log_message('################### END ###################')

        # tecnica 2
        #features = []
        #for seq_num in range(len(embedding_repr)):
        #    seq_len = (attention_mask[seq_num] == 1).sum()
        #    seq_emd = embedding_repr[seq_num][:seq_len - 1]
        #    features.append(torch.mean(seq_emd.cpu(), axis=0).numpy())

        return results #[results, features]

    #***********************************************************************************************
    # Configuration
    #***********************************************************************************************
    # In the following you can define your desired output. Current options:
    # per_residue embeddings
    # per_protein embeddings
    # secondary structure predictions

    if cfg["EMBEDDINGS"]["createflag"] == "no":
        logger.log_message("Embeddings already created. Skipping...")
        return

    per_residue = cfg["EMBEDDINGS"]["per_residue"] == 'True'
    #per_residue_path = outputFolder + "per_residue_embeddings.csv"  # where to store the embeddings

    # whether to retrieve per-protein embeddings
    # --> only one 1024-d vector per protein, irrespective of its length
    per_protein = cfg["EMBEDDINGS"]["per_protein"] == 'True'
    #per_protein_path = outputFolder + "per_protein_embeddings.csv"  # where to store the embeddings

    # whether to retrieve secondary structure predictions
    # This can be replaced by your method after being trained on ProtT5 embeddings
    sec_struct = cfg["EMBEDDINGS"]["sec_struct"] == 'True'
    #sec_struct_path = outputFolder + "sec_struct_preds.fasta"  # file for storing predictions

    # make sure that either per-residue or per-protein embeddings are stored
    assert per_protein is True or per_residue is True or sec_struct is True, logger.log_message(
        "Minimally, you need to activate per_residue, per_protein or sec_struct. (or any combination)")

    #device = torch.device("mps", 0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else cfg["ENVIRONMENT"]["torchdevice"])

    logger.log_message("Using {}".format(device))

    # Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
    logger.log_message("Loading T5 encoder model ...")
    model, tokenizer = get_T5_model(cfg, device)

    #***********************************************************************************************
    val_folder = os.path.join(cfg["UNIPROT"]["go_folder"], "downloads")
    sequencesBatchSize = cfg["EMBEDDINGS"]["sequencebatchsize"]
    #***********************************************************************************************

    outputFolder = os.path.join(cfg["UNIPROT"]["go_folder"], "embeddings")
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    dfCache = pd.DataFrame()
    if len(cfg["EMBEDDINGS"]["cachedataset"]) > 0:
        logger.log_message("Loading embeddings cache datasets ...")
        # Open the cache file in Original Input/Terrabacteria/Terrabacteria_embeddings_dataset.csv
        #if file exists, read it into dfCache
        for file in cfg["EMBEDDINGS"]["cachedataset"]:
            if os.path.exists(file):
                dftmp = pd.read_csv(file, delimiter=',', low_memory=False)
                dfCache = pd.concat([dfCache, dftmp], ignore_index=True)

    csv_files = [file for file in os.listdir(val_folder) if file.endswith('.dataset.csv')]
    csvCount = len(csv_files)
    #for each file in the folder whose name end with ".tsv"
    for idx, file in enumerate(csv_files):
        if file.endswith(".csv"):
            logger.log_message(f"\nProcessing file {file}\n", idx/csvCount)

            #define final_output_path equal to the outputFolder + the name of the original filename + "_embeddings.csv"
            final_output_path = os.path.join(outputFolder, file.split('.')[0] + ".embeddings.csv")
            f = open(final_output_path, 'w')

            #read the file into the first colum of df_si
            df = pd.read_csv(os.path.join(val_folder, file), delimiter=',')
            #Add a Specie column that is equal to the first two words of the Organism column
            df['Species'] = df['Organism'].apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
            #remove duplicates ffrom df
            df.drop_duplicates(subset=['Entry'], keep='first', inplace=True)

            logger.log_message (f"File has {len(df)} sequences ...")

            if len(cfg["EMBEDDINGS"]["cachedataset"]) > 0:

                cachedEntries = df[df['Entry'].isin(dfCache['Entry'])]
                cols_to_copy = list(range(1024))
                cols_to_copy = [str(x) for x in cols_to_copy]
                merged_df = cachedEntries.merge(dfCache[['Entry'] + cols_to_copy], how='left', on='Entry')
                cachedEntries = merged_df[merged_df.columns]

                #save in a new df cachedEntries the list of rows of df that are also in dfCache
                #cachedEntries = df[df['Entry'].isin(dfCache['Entry'])]
                #create in cachedEntries 1024 new columns called '0' to '1023' (strings)
                #cachedEntries = cachedEntries.reindex(columns=dfCache.columns)

                #For each row in cachedEntries
                # Create a mapping for efficient lookup
                #entry_mapping = dict(zip(dfCache['Entry'], dfCache.index))
                # Update cachedEntries directly using vectorized operations
                # count = 0
                # for index, row in cachedEntries.iterrows():
                #     matching_index = entry_mapping.get(row['Entry'])
                #     if matching_index is not None:
                #         cachedEntries.loc[index, '0':'1023'] = dfCache.loc[matching_index, '0':'1023']
                #         print(count)
                #         count = count + 1

                # for index, row in cachedEntries.iterrows():
                #     #find the corresponding row in dfCache
                #     matching_row = dfCache[dfCache['Entry'] == row['Entry']]
                #     #append the columns of matching_row named '0' to '1023' (strings) to the corresponding row in cachedEntries
                #     cachedEntries.loc[index, '0':'1023'] = matching_row.iloc[0]['0':'1023']
                logger.log_message (f"File has {len(cachedEntries)} sequences in cache ...")

                #remove from df all the rows that are cachedEntries
                df = df[~df['Entry'].isin(cachedEntries['Entry'])]
                logger.log_message (f"File has {len(df)} sequences not in cache ...")
            else:
                cachedEntries = pd.DataFrame()

            logger.log_message (f"Calculating embeddings for {len(df)} sequences ...")

            seqs = {}
            #Fill the seqs dict with the Entry Column as key and the Sequence column as value
            for index, row in df.iterrows():
                uniprot_id = row['Entry']
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")

                # repl. all white-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq = ''.join(row['Sequence'].split()).upper().replace("-", "")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                seqs[uniprot_id] = seq

            # Create a loop that calls the get_embedding function on batches of 25 seq proteins from df
            batchCount = 0
            logger.log_message (f"Building {(len(df)//sequencesBatchSize) + 1} batches of {sequencesBatchSize} sequences each");
            total_elements = len(df)

            for start_index in range(0, total_elements, sequencesBatchSize):
                end_index = min(start_index + sequencesBatchSize, total_elements)
                logger.log_message (f"\nProcessing batch {batchCount+1} of {(len(df)//sequencesBatchSize)+1} from {start_index} to {end_index} ...", (batchCount+1)/((len(df)//sequencesBatchSize)+1))
                batchCount += 1
                seqs_batch = {k: v for k, v in list(seqs.items())[start_index:end_index]}
                results = get_embeddings(model, tokenizer, seqs_batch, per_residue, per_protein, sec_struct)
                ft_pd = pd.DataFrame.from_dict(results["protein_embs"])
                ft_pd = ft_pd.T.reset_index()
                #Import the 1024 columns of ft_pd of each row in the corresponding row of df.
                #The join has to be made on df['Entry'] and ft_pd['index']
                #Do not import the column index from ft_pd
                #Rows of df that do not match have to remain in df
                #and the corresponding columns in ft_pd have to be filled with 0
                # Iterate over each row in ft_pd and update corresponding rows in df
                for _, row in ft_pd.iterrows():
                    entry_value = row['index']
                    matching_row = df[df['Entry'] == entry_value]
                    if not matching_row.empty:
                        df.loc[matching_row.index, row.index[1:]] = row.iloc[1:].values
                        #write the row matching_row to csv file f using the to_csv method
                        #with the parameters index=False, mode='w', header=f.tell() == 0
                        df.loc[matching_row.index].to_csv(f, index=False, mode='a', header=f.tell() == 0)


    #saved chachedEntries to the f file
    if len(cachedEntries) > 0:
        cachedEntries.to_csv(f, index=False, mode='a', header=f.tell() == 0)

    return

