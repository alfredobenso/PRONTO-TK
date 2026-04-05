
# 
# This script compares gene rankings obtained by differential transcription and
# differential translation analysis.
#
# Output files: 
# - ../results/DTL/tr.tl.rank.bin.tiff
#

require(xlsx)
require(dplyr)
require(ggpubr)
require(stringr)
require(gridExtra)
require(grid)
require(ComplexHeatmap)
require(colorRamp2)


isEmpty <- function(x) {
  return(length(x)==0)
}


#-------------------------------------------------------------------------------
# Load functional annotations
#-------------------------------------------------------------------------------

# Retrieve the NCBI annotations of the CDS in C.a' genome
ncbif = read.delim("../data/GCA_040166795.1/genomic.gff",
                   header=FALSE, comment.char = "#", sep="\t")
annotf = ncbif[ncbif[,3] %in% "CDS",]

# Extract gene ids
locusTags = as.vector(sapply(annotf[,9], function(x) strsplit(x,";")[[1]][2]))
# Remove unnecessary text from gene ids
locusTags = gsub("Parent=gene-", "", locusTags)
locusTags = gsub("Name=", "", locusTags)



#-------------------------------------------------------------------------------
# Load genes ranked according to DTR analysis
#-------------------------------------------------------------------------------

load("../results/DTR/totrna.de.rda")

# Remove genes external to C. autoethanogenum
totrna.de = totrna.de[grepl("ERCC",rownames(totrna.de))==FALSE,]

# Remove genes that are not protein-coding 
totrna.de = totrna.de[is.na(match(rownames(totrna.de),locusTags))==FALSE,]

# Order the data frame 
tmp = totrna.de$PValue
totrna.ord = totrna.de[order(tmp),]

# Identify genes whose ranks lie in the top 5% positions 
top_n = round( 0.05 * dim(totrna.ord)[1] )
top_n_totdeg = rownames(totrna.ord)[1:top_n]

#-------------------------------------------------------------------------------
# Load genes ranked according to metric used in DTL analysis 
#-------------------------------------------------------------------------------

# Load metrics used in DTL analysis
f.abs = read.delim("../results/DTL/pol.rnk/abs_translation_diff_results.csv", sep=",")
f.boot = read.delim("../results/DTL/pol.rnk/bootstrap_results.csv", sep=",")
f.wa = read.delim("../results/DTL/pol.rnk/weighted_aitchison_results.csv", sep=",")

# Remove ERCC genes used for control
f.abs = f.abs[grepl("ERCC",f.abs$X)==FALSE,]
f.boot = f.boot[grepl("ERCC",f.boot$gene)==FALSE,]
f.wa = f.wa[grepl("ERCC",f.wa$gene)==FALSE,]

# Remove genes that are not protein-coding 
f.abs = f.abs[is.na(match(f.abs$X,locusTags))==FALSE,]
f.boot = f.boot[is.na(match(f.boot$gene,locusTags))==FALSE,]
f.wa = f.wa[is.na(match(f.wa$gene,locusTags))==FALSE,]

genes = unique(c(f.abs[,1], f.boot[,2], f.wa[,2]))

#
# Retrieve gene ranks 
#

rnk.abs = c()
round.vec = round(f.abs$abs_diff, digits=4)
unique.round.vec = unique(round.vec)
sort.unique.round.vec = unique.round.vec[rev(order(unique.round.vec))]
for(i in 1 : length(genes)){ 
  idx = match(genes[i], f.abs$X)
  rnk.abs[i] = match(round(as.numeric(f.abs$abs_diff[idx]), digits=4), sort.unique.round.vec)
}
# normalize gene ranks 
rnk.abs.norm = 100 * rnk.abs/max(rnk.abs)
names(rnk.abs.norm) = genes


rnk.boot = c()
round.vec = c()
for(i in 1 : length(f.boot$gene)){
  round.vec[i] = paste(f.boot$num_sig_phases[i], round(f.boot$max_diff[i], digits=4), sep=";")
}
unique.round.vec = unique(round.vec)
sort.unique.round.vec = unique.round.vec[rev(order(unique.round.vec))]
for(i in 1 : length(genes)){
  idx = match(genes[i], f.boot$gene)
  tmp = paste(f.boot$num_sig_phases[idx], round(f.boot$max_diff[idx], digits=4), sep=";")
  rnk.boot[i] = match(tmp, sort.unique.round.vec)
}
# normalize gene ranks 
rnk.boot.norm = 100 * rnk.boot/max(rnk.boot)
names(rnk.boot.norm) = genes


rnk.wa = c()
round.vec = c()
for(i in 1 : length(f.wa$gene)){
  round.vec[i] = round(f.wa$weighted_distance[i], digits=4)
}
unique.round.vec = unique(round.vec)
sort.unique.round.vec = unique.round.vec[rev(order(unique.round.vec))]
for(i in 1 : length(genes)){
  idx = match(genes[i], f.wa$gene)
  rnk.wa[i] = match(round(as.numeric(f.wa$weighted_distance[idx]), digits=4), sort.unique.round.vec)
}
# normalize gene ranks 
rnk.wa.norm = 100 * rnk.wa/max(rnk.wa)
names(rnk.wa.norm) = genes


# Compute the average rank for each gene
rnk.mat = as.matrix(cbind.data.frame(rnk.abs.norm, rnk.boot.norm, rnk.wa.norm))
rownames(rnk.mat) = genes
# arithmetic mean
rnk.mean = apply(rnk.mat,1,mean) 
rnk.mean.sorted = rnk.mean[order(as.numeric(rnk.mean))]
# save the result for subsequent usage
pol.rnk = rnk.mean.sorted


# Retrieve common genes to the two analyses by RNA-seq and Pol-seq
commonGene = intersect(names(pol.rnk), rownames(totrna.de))

# Retrieve gene ranks according to the two analyses for common genes
totrna.de.rnk = 100*match(commonGene, rownames(totrna.ord))/length(commonGene)
pol.de.rnk = rnk.mean.sorted[match(commonGene, names(rnk.mean.sorted))]

# Combine gene ranks according to the two analyses
de.rnk = cbind.data.frame(totrna.de.rnk, pol.de.rnk)
rownames(de.rnk) = commonGene


##-------------------------------------------------
##
## Plot the gene counts by bins of ranks according 
## to TR and TL analyses in a heatmap. 
## Ranks are normalized in the range [0,100].
##
##-------------------------------------------------

subset_1_pol <- subset(names(pol.de.rnk), pol.de.rnk>=0 & pol.de.rnk<10)
subset_2_pol <- subset(names(pol.de.rnk), pol.de.rnk>=10 & pol.de.rnk<20)
subset_3_pol <- subset(names(pol.de.rnk), pol.de.rnk>=20 & pol.de.rnk<30)
subset_4_pol <- subset(names(pol.de.rnk), pol.de.rnk>=30 & pol.de.rnk<40)
subset_5_pol <- subset(names(pol.de.rnk), pol.de.rnk>=40 & pol.de.rnk<50)
subset_6_pol <- subset(names(pol.de.rnk), pol.de.rnk>=50 & pol.de.rnk<60)
subset_7_pol <- subset(names(pol.de.rnk), pol.de.rnk>=60 & pol.de.rnk<70)
subset_8_pol <- subset(names(pol.de.rnk), pol.de.rnk>=70 & pol.de.rnk<80)
subset_9_pol <- subset(names(pol.de.rnk), pol.de.rnk>=80 & pol.de.rnk<90)
subset_10_pol <- subset(names(pol.de.rnk), pol.de.rnk>=90 & pol.de.rnk<100)

subset_pol_list = list(subset_1_pol,
                       subset_2_pol,
                       subset_3_pol,
                       subset_4_pol,
                       subset_5_pol,
                       subset_6_pol,
                       subset_7_pol,
                       subset_8_pol,
                       subset_9_pol,
                       subset_1_pol,
                       subset_2_pol,
                       subset_10_pol
                       )

subset_1_tot <- subset(commonGene, totrna.de.rnk>=0 & totrna.de.rnk<10)
subset_2_tot <- subset(commonGene, totrna.de.rnk>=10 & totrna.de.rnk<20)
subset_3_tot <- subset(commonGene, totrna.de.rnk>=20 & totrna.de.rnk<30)
subset_4_tot <- subset(commonGene, totrna.de.rnk>=30 & totrna.de.rnk<40)
subset_5_tot <- subset(commonGene, totrna.de.rnk>=40 & totrna.de.rnk<50)
subset_6_tot <- subset(commonGene, totrna.de.rnk>=50 & totrna.de.rnk<60)
subset_7_tot <- subset(commonGene, totrna.de.rnk>=60 & totrna.de.rnk<70)
subset_8_tot <- subset(commonGene, totrna.de.rnk>=70 & totrna.de.rnk<80)
subset_9_tot <- subset(commonGene, totrna.de.rnk>=80 & totrna.de.rnk<90)
subset_10_tot <- subset(commonGene, totrna.de.rnk>=90 & totrna.de.rnk<100)

subset_tot_list = list(subset_1_tot,
                       subset_2_tot,
                       subset_3_tot,
                       subset_4_tot,
                       subset_5_tot,
                       subset_6_tot,
                       subset_7_tot,
                       subset_8_tot,
                       subset_9_tot,
                       subset_1_tot,
                       subset_2_tot,
                       subset_10_tot
)

m=matrix(0, nrow=10, ncol=10, dimnames=list(
  c("[0,10[","[10,20[","[20,30[","[30,40[","[40,50[","[50,60[","[60,70[","[70,80]","[80,90[","[90,100]"),
  c("[0,10[","[10,20[","[20,30[","[30,40[","[40,50[","[50,60[","[60,70[","[70,80]","[80,90[","[90,100]")
  )
)

for(i in 1 : 10){
  for(j in 1 : 10){
    m[i,j] = length(intersect(subset_pol_list[[i]], subset_tot_list[[j]]))
  }  
}


tiff("tr.tl.rank.bin.tiff",res=600, width=3000, height=3000)
Heatmap(m, cluster_rows = FALSE, cluster_columns = FALSE, 
        rect_gp = gpar(col = "white", lwd = 2),
        column_title = "Rank bin by TR", row_title = "Rank bin by TL", 
        heatmap_legend_param = list(
          title = "Count", row_labels = rownames(m), column_labels = rownames(m)),
        column_title_gp = gpar(cex=1.5), row_title_gp = gpar(cex=1.5)
)
dev.off()


