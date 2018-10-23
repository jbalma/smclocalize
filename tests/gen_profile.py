import pstats
prof_name='./smcloc_profile.prof'
#p = pstats.Stats(prof_name, stream=output)
#p = pstats.Stats('restats')
#p.strip_dirs().sort_stats(-1)p.strip_dirs().sort_stats(-1)
#p.sort_stats('time', 'cum').print_stats(.5, 'init')

with open('mystats_output.txt', 'wt') as output:
    #stats = Stats(prof_name, stream=output)
    p = pstats.Stats(prof_name, stream=output)
    p.sort_stats('cumulative', 'tottime')
    p.
    p.strip_dirs()
    p.print_stats()
