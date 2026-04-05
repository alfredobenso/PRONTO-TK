
##
## This script carries out two approaches to compare the distributions of gene 
## ranks across the three metrics we used to evaluate translational regulation. 
##
## Approach A: Gene ranks were normalized to the range [0, 100] according to 
## each of the metrics that we used to evaluate translational regulation 
## (TL-STATUS, TL-PROFILE, and POLY-FRACTIONS). Firstly, we determined the 
## normalized ranks that the top 5% ranking genes by a certain metric obtained 
## by using another metric. Secondly, we evaluated if this rank distribution is 
## significantly different from a rank distribution obtained by random sampling of gene ranks 
## without replacement from a uniform distribution in the interval [0,100]. For this, 
## we used both the Mann-Whitney (MW) test, which detects differences in central tendency, 
## and the Kolmogorov-Smirnov (KS) test, which is sensitive to any difference in the 
## distribution shapes. Since the statistical significance of the test result can 
## vary according to the randomly sampled rank distribution, we repeated both tests 10^6 times 
## to assess their range of variability. We carried out his procedure for each pair 
## of the three metrics. 
##
## Approach B: Identical to Approach A, we normalizsed gene ranks to the range 
## [0, 100] according to each of the metrics that we used to evaluate translational 
## regulation (TL-STATUS, TL-PROFILE, and POLY-FRACTIONS). Firstly, we determined 
## the normalized ranks that the top 5% ranking genes by a certain metric obtain 
## by using another metric. Secondly, we considered the median value of these 
## ranks for each metric as the test statistic and evaluated if it significantly 
## differs from random expectations. For this, we constructed the null distribution 
## of the test statistic by computing its value for each of the 10^6 random rank 
## distributions that were obtained by random sampling a number of ranks equal to 
## the number of top 5% ranking genes from a uniform distribution on the interval [0, 100]. 
## We carried out this procedure for each pair of the three metrics. 
## The empirical p-value was then obtained as the fraction of cases that produce a 
## test statistic at least as low as that calculated for experimental data
##
## Output files: 
## - ../results/DTL/consistency.dtl.dtr.vA.multiple.png
## - ../results/DTL/consistency.dtl.dtr.vA.single.png
## - ../results/DTL/consistency.dtl.dtr.vB.png
##

require(openxlsx)
require(ggplot2)
require(plyr)
require(patchwork)
require(cowplot)


#-------------------------------------------------------------------------------
# Load functional annotations
#-------------------------------------------------------------------------------
ncbif = read.delim("../data/GCA_040166795.1/genomic.gff",
                   header=FALSE, comment.char = "#", sep="\t")
annotf = ncbif[ncbif[,3] %in% "CDS",]

# Extract gene ids
locusTags = as.vector(sapply(annotf[,9], function(x) strsplit(x,";")[[1]][2]))
# Remove unnecessary text from gene ids
locusTags = gsub("Parent=gene-", "", locusTags)
locusTags = gsub("Name=", "", locusTags)


#-------------------------------------------------------------------------------
# Load genes ranked according to metric used in DTL analysis 
#-------------------------------------------------------------------------------

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

# TL-STATUS approach
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

# POLY-FRACTIONS approach
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

# TL-PROFILE approach
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

#
# Retrieve top-ranking genes by each metric
#

# POLY-FRACTIONS approach
boot.round.un = unique(round(rnk.boot.norm, digits=4))
boot.round.un.ord = boot.round.un[order(boot.round.un)]
idx = round( 0.05*length(boot.round.un.ord) )
rnk.th = boot.round.un.ord[idx]
top_n = length(rnk.boot.norm[rnk.boot.norm<rnk.th])
top.n.boot = names(rnk.boot.norm[rnk.boot.norm<rnk.th])

# TL-STATUS approach
abs.round.un = unique(round(rnk.abs.norm, digits=4))
abs.round.un.ord = abs.round.un[order(abs.round.un)]
idx = round( 0.05*length(abs.round.un.ord) )
rnk.th = abs.round.un.ord[idx]
top_n = length(rnk.abs.norm[rnk.abs.norm<rnk.th])
top.n.abs = names(rnk.abs.norm[rnk.abs.norm<rnk.th] )

# TL-PROFILE approach
wa.round.un = unique(round(rnk.wa.norm, digits=4))
wa.round.un.ord = wa.round.un[order(wa.round.un)]
idx = round( 0.05*length(wa.round.un.ord) )
rnk.th = wa.round.un.ord[idx]
top_n = length(rnk.wa.norm[rnk.wa.norm<rnk.th])
top.n.wa = names(rnk.wa.norm[rnk.wa.norm<rnk.th])


#-------------------
# Approach A 
#-------------------

##
## Plot the results of a single KS test and MN test used in Approach A to show the data underlying a test
##

png("../results/DTL/consistency.dtl.meths.vA.single.png", res=600, width=9000, height= 2000)

#
# TL-PROFILE vs TL-STATUS
#
real.pos = rnk.abs.norm[match(top.n.wa,names(rnk.abs.norm))]
rand.pos = runif(length(top.n.wa), 0, 100)

# Using two-sample Kolmogorov-Smirnov test
tl.profile.vs.tl.status.ks.test = ks.test(real.pos, rand.pos, alternative="greater")

df <- data.frame(values = c(real.pos, 
                            rand.pos
                            ),
                Data = c(rep("Real", length(real.pos)),
                         rep("Bootstrap", length(rand.pos))
                         )
                )

p1 <- ggplot(df, aes(values, colour = Data)) +
  stat_ecdf(geom = "step")  +
  theme_light() +
  theme(text = element_text(size = 11.5)) +
  xlab("Rank") +
  ylab("Ecdf") +
  labs(title = "TL-PROFILE vs. TL-STATUS") + theme(title = element_text(size = 8.5)) + 
  scale_color_manual(values=c("skyblue","darkblue")) +
  scale_fill_manual(values=c("skyblue","darkblue"))


p2 <- ggplot(df, aes(Data, values)) +
  geom_boxplot(, notch = TRUE, alpha = 0.5, fill = c("skyblue","darkblue")) + #+ geom_jitter(width = 0.2) +
  theme_light() +
  theme(text = element_text(size = 11.5)) +
  xlab("") +
  ylab("Rank") +
  labs(title = "")

# Using Wilcoxon rank-sum test or Mann-Whitney test
tl.profile.vs.tl.status.wilcox.test = wilcox.test(real.pos, rand.pos, 
                                                  paired = FALSE, alternative = "less")


#
# TL-PROFILE vs POLY-FRACTIONS
#
real.pos = rnk.boot.norm[match(top.n.wa,names(rnk.boot.norm))]
rand.pos = runif(length(top.n.wa), 0, 100)

# Using two-sample Kolmogorov-Smirnov test
tl.profile.vs.poly.fractions.ks.test = ks.test(real.pos, rand.pos, alternative="greater")

df <- data.frame(values = c(real.pos, 
                            rand.pos
                            ),
                Data = c(rep("Real", length(real.pos)),
                         rep("Bootstrap", length(rand.pos)
                             )
                         )
                )

p3 <- ggplot(df, aes(values, colour = Data)) +
  stat_ecdf(geom = "step")  +
  theme_light() +
  theme(text = element_text(size = 11.5)) +
  xlab("Rank") +
  ylab("Ecdf") +
  labs(title = "TL-PROFILE vs. POLY-FRACTIONS") + theme(title = element_text(size = 8.5)) + 
  scale_color_manual(values=c("skyblue","darkblue")) +
  scale_fill_manual(values=c("skyblue","darkblue"))


p4 <- ggplot(df, aes(Data, values)) +
  geom_boxplot(, notch = TRUE, alpha = 0.5, fill = c("skyblue","darkblue")) + #+ geom_jitter(width = 0.2) +
  theme_light() +
  theme(text = element_text(size = 11.5)) +
  xlab("") +
  ylab("Rank") +
  labs(title = "")

# Using Wilcoxon rank-sum test or Mann-Whitney test
tl.profile.vs.poly.fractions.wilcox.test = wilcox.test(real.pos, rand.pos, 
                                                       paired = FALSE, alternative = "less")

#
# TL-STATUS vs POLY-FRACTIONS
#
real.pos = rnk.boot.norm[match(top.n.abs,names(rnk.boot.norm))]
rand.pos = runif(length(top.n.abs), 0, 100)

# Using two-sample Kolmogorov-Smirnov test
tl.status.vs.poly.fractions.ks.test = ks.test(real.pos, rand.pos, alternative="greater")

df <- data.frame(values = c(real.pos, 
                            rand.pos
                            ),
                Data = c(rep("Real", length(real.pos)),
                         rep("Bootstrap", length(rand.pos)
                             )
                         )
                )

p5 <- ggplot(df, aes(values, colour = Data)) +
  stat_ecdf(geom = "step")  +
  theme_light() +
  theme(text = element_text(size = 11.5)) +
  xlab("Rank") +
  ylab("Ecdf") +
  labs(title = "TL-STATUS vs. POLY-FRACTIONS") + theme(title = element_text(size = 8.5)) + 
  scale_color_manual(values=c("skyblue","darkblue")) +
  scale_fill_manual(values=c("skyblue","darkblue"))


p6 <- ggplot(df, aes(Data, values)) +
  geom_boxplot(, notch = TRUE, alpha = 0.5, fill = c("skyblue","darkblue")) + #+ geom_jitter(width = 0.2) +
  theme_light() +
  theme(text = element_text(size = 11.5)) +
  xlab("") +
  ylab("Rank") +
  labs(title = "")


# Using Wilcoxon rank-sum test or Mann-Whitney test
tl.status.vs.poly.fractions.wilcox.test = wilcox.test(real.pos, rand.pos, 
                                                      paired = FALSE, alternative = "less")

plot_grid(p1, p2, p3, p4, p5, p6, ncol=6, rel_heights = c(.8, 1, .8, 1, .8, 1), 
          rel_widths = c(.8, .5, .8, .5, .8, .5))

dev.off()

##
## Plot the outcomes of multiple tests according to Approach A
##

tiff("../results/DTL/consistency.dtl.meths.vA.multiple.tiff", res=600, width=5500, height= 1500)

# TL-PROFILE vs TL-STATUS
real.pos = rnk.abs.norm[match(top.n.wa,names(rnk.abs.norm))]
N=1.0E06
ks.pvs = c()
wilcox.pvs = c()
for(i in 1 : N){
  rand.pos = runif(length(top.n.wa), 0, 100)
  ks.pvs[i] = -log10(ks.test(real.pos, rand.pos, alternative="greater")$p.value)
  wilcox.pvs[i] = -log10(wilcox.test(real.pos, rand.pos, alternative="less")$p.value)
  show(i)
}

df <- data.frame(values = c(ks.pvs, 
                            wilcox.pvs
                            ),
                Test = c(rep("KS", length(ks.pvs)),
                         rep("MW", length(wilcox.pvs))
                         )
                )

p1 <- ggplot(df, aes(x = values, fill=Test)) +
  geom_histogram(position = "identity", alpha = 0.4, bins = 30) +
  scale_fill_manual(values=c("#E69F00", "#998ec3")) + 
  theme_light() +
  theme(text = element_text(size = 9.5)) + 
  xlab(expression(-log[10](p-value))) +
  ylab("Count") +
  labs(title="TL-PROFILE vs. TL-STATUS")


# TL-PROFILE vs POLY-FRACTIONS
real.pos = rnk.boot.norm[match(top.n.wa,names(rnk.boot.norm))]
N=1.0E06
ks.pvs = c()
wilcox.pvs = c()
for(i in 1 : N){
  rand.pos = runif(length(top.n.wa), 0, 100)
  ks.pvs[i] = -log10(ks.test(real.pos, rand.pos, alternative="greater")$p.value)
  wilcox.pvs[i] = -log10(wilcox.test(real.pos, rand.pos, alternative="less")$p.value)
  show(i)
}

df <- data.frame(values = c(ks.pvs, 
                            wilcox.pvs
                            ),
                Test = c(rep("KS", length(ks.pvs)),
                         rep("MW", length(wilcox.pvs))
                         )
                )

p2 <- ggplot(df, aes(x = values, fill=Test)) +
  geom_histogram(position = "identity", alpha = 0.4, bins = 30) +
  scale_fill_manual(values=c("#E69F00", "#998ec3")) + 
  theme_light() +
  theme(text = element_text(size = 9.5)) + 
  xlab(expression(-log[10](p-value))) +
  ylab("Count") +
  labs(title="TL-PROFILE vs. POLY-FRACTIONS")


# TL-STATUS vs POLY-FRACTIONS
real.pos = rnk.boot.norm[match(top.n.abs,names(rnk.boot.norm))]
N=1.0E06
ks.pvs = c()
wilcox.pvs = c()
for(i in 1 : N){
  rand.pos = runif(length(top.n.abs), 0, 100)
  ks.pvs[i] = -log10(ks.test(real.pos, rand.pos, alternative="greater")$p.value)
  wilcox.pvs[i] = -log10(wilcox.test(real.pos, rand.pos, alternative="less")$p.value)
  show(i)
}


df <- data.frame(values = c(ks.pvs, 
                            wilcox.pvs
                            ),
                  Test = c(rep("KS", length(ks.pvs)),
                           rep("MW", length(wilcox.pvs))
                           )
                 )

p3 <- ggplot(df, aes(x = values, fill=Test)) +
  geom_histogram(position = "identity", alpha = 0.4, bins = 30) +
  scale_fill_manual(values=c("#E69F00", "#998ec3")) + 
  theme_light() +
  theme(text = element_text(size = 11.5)) + 
  xlab(expression(-log[10](p-value))) +
  ylab("Count") +
  labs(title="TL-STATUS vs. POLY-FRACTIONS")

p1 + p2 + p3

dev.off()



#-----------------
# Approach B
#-----------------

png("../results/DTL/consistency.dtl.meths.vB.png", res=600, width = 2500, height = 5500)

#
# TL-PROFILE vs TL-STATUS
#
N = 1E06
real.rnk = rnk.abs.norm[match(top.n.wa, names(rnk.abs.norm))]
real.rnk.median.wa.vs.abs = median(real.rnk)
rand.rnk.median = c()
for(i in 1 : N){
  rand.rnk = runif(length(top.n.wa), 0, 100)
  rand.rnk.median[i] = median(rand.rnk)
}

df <- data.frame(values = rand.rnk.median)

p1 <- ggplot(df, aes(x = values)) +
  geom_histogram(position = "identity", alpha = 0.4, fill = "skyblue", bins = 30) +
  theme_light() +
  theme(axis.title = element_text(size = 12.5), 
        axis.text = element_text(size = 10.5),
        title = element_text(size = 10.5)
  ) + 
  xlab("Median rank") +
  ylab("Count") +
  labs(title = "TL-PROFILE vs. TL-STATUS") +
  geom_vline(aes(xintercept=real.rnk.median.wa.vs.abs), colour="darkblue",
             linetype="dashed", linewidth=.95) 


# TL-PROFILE vs POLY-FRACTIONS 
N = 1E06
real.rnk = rnk.boot.norm[match(top.n.wa, names(rnk.boot.norm))]
real.rnk.median.wa.vs.boot = median(real.rnk)
rand.rnk.median = c()
for(i in 1 : N){
  rand.rnk = runif(length(top.n.wa), 0, 100)
  rand.rnk.median[i] = median(rand.rnk)
}


df <- data.frame(values = rand.rnk.median)

p2 <- ggplot(df, aes(x = values)) +
  geom_histogram(position = "identity", alpha = 0.4, fill = "skyblue", bins = 30) +
  theme_light() +
  theme(axis.title = element_text(size = 12.5), 
        axis.text = element_text(size = 10.5),
        title = element_text(size = 10.5)
  ) + 
  xlab("Median rank") +
  ylab("Count") +
  labs(title = "TL-PROFILE vs. POLY-FRACTIONS") +
  geom_vline(aes(xintercept=real.rnk.median.wa.vs.boot), colour="darkblue",
             linetype="dashed", linewidth=.95) 

# TL-STATUS vs POLY-FRACTIONS
N = 1E06
real.rnk = rnk.boot.norm[match(top.n.abs, names(rnk.boot.norm))]
real.rnk.median.abs.vs.boot = median(real.rnk)
rand.rnk.median = c()
for(i in 1 : N){
  rand.rnk = runif(length(top.n.abs), 0, 100)
  rand.rnk.median[i] = median(rand.rnk)
}


df <- data.frame(values = rand.rnk.median)

p3 <- ggplot(df, aes(x = values)) +
  geom_histogram(position = "identity", alpha = 0.4, fill = "skyblue", bins = 30) +
  theme_light() +
  theme(axis.title = element_text(size = 12.5), 
        axis.text = element_text(size = 10.5),
        title = element_text(size = 10.5)
        ) + 
  xlab("Median rank") +
  ylab("Count") +
  labs(title = "TL-STATUS vs. POLY-FRACTIONS") +
  geom_vline(aes(xintercept=real.rnk.median.abs.vs.boot), colour="darkblue",
             linetype="dashed", linewidth=.95) 

combined_plot <- p1 + p2 + p3 + plot_layout(ncol = 1)

combined_plot 

dev.off()


