from goatools.base import get_godag
from goatools.godag.go_tasks import get_go2descendants
import os.path
import requests
import re
import pandas as pd
import io
import numpy as np
import re
import requests
from requests.adapters import HTTPAdapter, Retry

def get_GODescendants(go_ids):
  godag = get_godag("go-basic.obo", optional_attrs={'relationship'})
  go_terms = [godag[go_id] for go_id in go_ids]
  optional_relationships = {'is_a'}
  all_descendants = get_go2descendants(go_terms, optional_relationships)
  return all_descendants


def downloadUPProteins(cfg=None, logger=None):

  def get_next_link(headers):
    if "Link" in headers:
      match = re_next_link.match(headers["Link"])
      if match:
        return match.group(1)

  def get_batch(batch_url):
    while batch_url:
      response = session.get(batch_url)
      response.raise_for_status()
      total = response.headers["x-total-results"]
      yield response, total
      batch_url = get_next_link(response.headers)

  def getBatch2File(url, nomefile, logger, nmax=None):
    if int(nmax) == -1:
      nmax = None

    progress = 0
    with open(nomefile, 'w') as f:
      for batch, total in get_batch(url):
        lines = batch.text.splitlines()
        for line in lines[1:]:
          print(line, file=f)
        progress += len(lines[1:])
        logger.log_message(f'{progress} / {total}')
        if nmax and progress >= int(nmax):
          break  # Stop iterating if NMAX entries are processed

  def getBatch2Df(url, df, logger, nmax=None):
    if int(nmax) == -1:
      nmax = None
    progress = 0
    for batch, total in get_batch(url):
      lines = batch.text.splitlines()
      # Skip the header line if it's already present in the DataFrame
      data = pd.DataFrame([line.split('\t') for line in lines[1:]], columns=lines[0].split('\t'))
      df = pd.concat([df, data], ignore_index=True)
      progress += len(lines[1:])
      logger.log_message(f'{progress} / {total}', float(progress)/float(total))
      if nmax and progress >= int(nmax):
        break  # Stop iterating if NMAX entries are processed

    return df

  def runUPqueries(cfg, label = "YES", logger = None):

    #Configuration
    if cfg["UNIPROT"]["go_includedescendants"] == "True":
      finalTerms = get_GODescendants(cfg["UNIPROT"]["go_ids"])
    else:
      finalTerms = cfg["UNIPROT"]["go_ids"]

    if label == "YES":
      go_reviewed = cfg["UNIPROT-Label_1"]["reviewed"] #["false", "true"]
      go_manual_auto = cfg["UNIPROT-Label_1"]["annotation"] #["manual", "automatic"]
    else:
      go_reviewed = cfg["UNIPROT-Label_0"]["reviewed"] #["false", "true"]
      go_manual_auto = cfg["UNIPROT-Label_0"]["annotation"] #["manual", "automatic"]

    #go_manual_auto1 is a list of strings that contains the terms in go_manual_auto adding "go_" at the beginning of each term if the word "go" is not present
    go_manual_auto = ["go_" + k if "go" not in k else k for k in go_manual_auto]

    totalLoops = len(go_reviewed) * len(go_manual_auto)
    loopCount = 0
    for rev in go_reviewed:
      for gotype in go_manual_auto:
        df = pd.DataFrame()
        nomefile = "Label_" + label + "_" + "_".join(cfg["UNIPROT"]["go_taxonomies"]) + "_" + rev + "_" + gotype + "_" + "_".join(cfg["UNIPROT"]["go_ids"]) + ".tsv"
        logger.log_message(f"\n\tLabel: {label} Taxonomies: {', '.join(cfg['UNIPROT']['go_taxonomies'])}, reviewed: {rev}, gotype: {gotype.replace('go_','')}, go_ids: {', '.join(finalTerms)}")

        format = "format=tsv"
        size = "size=" + str(cfg["UNIPROT"]["go_batchsize"])
        fields = "fields=accession,reviewed,id,protein_name,gene_names,organism_name,length,cc_caution,go_f,sequence"

        if cfg["UNIPROT"]["go_batchsize"] == -1:
          baseurl = f"https://rest.uniprot.org/uniprotkb/stream?{fields}&{format}&query=("
        else:
          baseurl = f"https://rest.uniprot.org/uniprotkb/search?{fields}&{size}&{format}&query=("

        q_rev = "(reviewed:" + rev + ")"
        b1 = ''.join(["OR+(taxonomy_id:" + str(i) + ")+" for i in cfg['UNIPROT']['go_taxonomies']])
        q_tax = "+AND+(" + b1[3:-1] + ")"
        b2 = ''.join(["OR+(" + gotype + ":" + str(i).replace("GO:","") + ")+" for i in finalTerms])
        q_gos = "+AND+(" + b2[3:-1] + ")"
        b3 = ''.join(["OR+(" + gotype + ":" + str(i).replace("GO:","") + ")+" for i in finalTerms])
        q_gos_NOT = "+NOT+(" + b3[3:-1] + ")"

        if label == "YES":
          url = baseurl + q_rev + q_tax + q_gos + ")"
        else:
          url = baseurl + q_rev + q_tax + q_gos_NOT + ")"

        logger.log_message(f'URL: {url}')

        if cfg["UNIPROT"]["go_batchsize"] == -1:
          logger.log_message(f'Downloading proteins ...')
          all_fastas = requests.get(url).text
          df = pd.read_csv(io.StringIO((all_fastas)), sep='\t', escapechar='\n')
        else:
          # Compute number of proteins returned by the query
          response = session.get(url)
          response.raise_for_status()
          total = response.headers["x-total-results"]
          logger.log_message(f'Total number of proteins for label: {label}: {total}')
          if int(cfg["UNIPROT"]["go_maxproteinsdownload"]) != -1:
            total = min(int(total), int(cfg["UNIPROT"]["go_maxproteinsdownload"]))
          logger.log_message(f'Number of proteins downloaded in each batch: {cfg["UNIPROT"]["go_batchsize"]}', loopCount / totalLoops)
          df = getBatch2Df(url, df, logger, total)

        if "Annotation" not in df.columns:
          df.insert(0, "Annotation", None)
        df["Annotation"].fillna(gotype.replace("go_", ""), inplace=True)

        if "Label" not in df.columns:
          df.insert(0, "Label", None)
          logger.log_message(f'Column "Label" added')

        df["Label"].fillna((1 if label == "YES" else 0), inplace=True)
        logger.log_message(f'Label set to {(1 if label == "YES" else 0)}')

        #if cfg["UNIPROT"]["go_folder"], "downloads" does not exist, create it
        if not os.path.exists(os.path.join(cfg["UNIPROT"]["go_folder"], "downloads")):
            os.makedirs(os.path.join(cfg["UNIPROT"]["go_folder"], "downloads"))
        df.to_csv(os.path.join(cfg["UNIPROT"]["go_folder"], "downloads", nomefile), index=False, sep='\t')

        loopCount += 1

    return df

  ###### main code of the function downloadUPProteins() ######
  if cfg["UNIPROT"]["createflag"].lower() != "no":
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))
    re_next_link = re.compile(r'<(.+)>; rel="next"')

    runUPqueries(cfg,"YES", logger = logger)
    runUPqueries(cfg,"NO", logger = logger)

  else:
    logger.log_message(f"Download skipped because the destination file {os.path.join(cfg['UNIPROT']['go_folder'], cfg['UNIPROT']['datasetname'] + '.dataset.csv')} is already present.")

  #Now I want to merge together all the df stored in the folder downloads and save the result in a single file
  jjDF = pd.DataFrame()
  for file in os.listdir(os.path.join(cfg["UNIPROT"]["go_folder"], "downloads")):
    if file.startswith("Label_") and file.endswith(".tsv"):
      df = pd.read_csv(os.path.join(cfg["UNIPROT"]["go_folder"], "downloads", file), sep='\t')
      jjDF = pd.concat([jjDF, df], ignore_index=True)

  # Sort the DataFrame so that 'manual' comes before 'automatic'
  jjDF = jjDF.sort_values('Annotation', ascending=False)

  # Now drop duplicates based on 'Entry', keeping the first occurrence (which will be 'manual' if it exists)
  jjDF = jjDF.drop_duplicates(subset='Entry', keep='first')

  #Make column Label of type int
  jjDF.to_csv(os.path.join(cfg["UNIPROT"]["go_folder"], "downloads", cfg["UNIPROT"]["datasetname"] + ".dataset.csv"), index=False, header=True)
  logger.log_message(f"Final dataset saved to {os.path.join(cfg['UNIPROT']['go_folder'], 'downloads', cfg['UNIPROT']['datasetname'] + '.dataset.csv')}")

if __name__ == "__main__":
  df = downloadUPProteins()
