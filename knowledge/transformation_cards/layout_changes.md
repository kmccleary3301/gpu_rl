# Layout Changes

Layout changes alter how the logical tensor maps onto memory so the access pattern matches the kernel. This is often the difference between a gather-heavy, cache-thrashing path and a coalesced path that can actually approach peak bandwidth.
