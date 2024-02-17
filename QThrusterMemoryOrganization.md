<span style="color:#3878B2"> __Memory __ </span>  <span style="color:#3878B2"> __Organization – By Branch__ </span>

<span style="color:#00B050">Example: </span>  <span style="color:#00B050"> _7_ </span>  <span style="color:#00B050"> particles in </span>  <span style="color:#00B050"> _4_ </span>  <span style="color:#00B050"> branches</span>

Parallelization must be done by  <span style="color:#0070C0"> __branch__ </span>

This is  <span style="color:#00B050"> __m__ </span>  <span style="color:#00B050"> __ore efficient __ </span> for calculating inter\-particle forces

Also requires  <span style="color:#C00000"> __more memory:__ </span>

<span style="color:#3878B2"> __Memory __ </span>  <span style="color:#3878B2"> __Organization – By Branch__ </span>

<span style="color:#00B050">Example: </span>  <span style="color:#00B050"> _7_ </span>  <span style="color:#00B050"> particles in </span>  <span style="color:#00B050"> _4_ </span>  <span style="color:#00B050"> branches</span>

Parallelization must be done by  <span style="color:#0070C0"> __branch__ </span>

This is  <span style="color:#00B050"> __m__ </span>  <span style="color:#00B050"> __ore efficient __ </span> for calculating inter\-particle forces

Also requires  <span style="color:#C00000"> __more memory:__ </span>

PARTICLE \# IN BRANCH

_0_  __    __  _1_  __    __  _2_  __ __  __   __  _3_  __    __  _4_  __ __  __   __  _5_  __ __  __   __  _6_  __ __  __   __  _7_  __    __  _8_  __    __  _9_  __   __  __ __  _10_  __   __  _11_

_0    1    2_  _ _  _   3_  _ _  _   0    _  _1_  _    _  _2_  _    3    0    1    2    3_

<span style="color:#00B0F0"> __0    0    __ </span>  <span style="color:#00B0F0"> __0__ </span>  <span style="color:#00B0F0"> __    __ </span>  <span style="color:#00B0F0"> __0__ </span>  <span style="color:#00B0F0"> __    __ </span>  <span style="color:#00B0F0"> __1__ </span>  <span style="color:#00B0F0"> __    __ </span>  <span style="color:#00B0F0"> __1__ </span>  <span style="color:#00B0F0"> __    1    __ </span>  <span style="color:#00B0F0"> __1__ </span>  <span style="color:#00B0F0"> __    2    __ </span>  <span style="color:#00B0F0"> __2__ </span>  <span style="color:#00B0F0"> __    2    2__ </span>

__\_          \_           \_           \_         __  __ __  __  \_  __  __ __  __       \_           \_           \_           \_          \_  __  __ __  __   __  __ __  __     \_           \___

<span style="color:#3878B2"> __Memory __ </span>  <span style="color:#3878B2"> __Organization – By Branch__ </span>

__On\-Chip Register Memory__

_0_  __    __  _1_  __    __  _2_  __ __  __   __  _3_  __    __  _4_  __ __  __   __  _5_  __ __  __   __  _6_  __ __  __   __  _7_  __    __  _8_  __    __  _9_  __   __  __ __  _10_  __   __  _11_

_0    1    2_  _ _  _   3_  _ _  _   0    _  _1_  _    _  _2_  _    3    0    1    2    3_

<span style="color:#00B0F0"> __0    0    __ </span>  <span style="color:#00B0F0"> __0__ </span>  <span style="color:#00B0F0"> __    __ </span>  <span style="color:#00B0F0"> __0__ </span>  <span style="color:#00B0F0"> __    __ </span>  <span style="color:#00B0F0"> __1__ </span>  <span style="color:#00B0F0"> __    __ </span>  <span style="color:#00B0F0"> __1__ </span>  <span style="color:#00B0F0"> __    1    __ </span>  <span style="color:#00B0F0"> __1__ </span>  <span style="color:#00B0F0"> __    2    __ </span>  <span style="color:#00B0F0"> __2__ </span>  <span style="color:#00B0F0"> __    2    2__ </span>

__\_          \_           \_           \_         __  __ __  __  \_  __  __ __  __       \_           \_           \_           \_          \_  __  __ __  __   __  __ __  __     \_           \___

<span style="color:#3878B2"> __Particles Crossing Cell Boundaries__ </span>

<span style="color:#00B050">Example: </span>  <span style="color:#00B050"> _6_ </span>  <span style="color:#00B050"> particles in </span>  <span style="color:#00B050"> _2_ </span>  <span style="color:#00B050"> branches</span>

When particles cross the boundaries of the branches they must be transferred to the appropriate location in memory

__Race conditions must be avoided__

<span style="color:#3878B2"> __Particles Crossing Cell Boundaries__ </span>

<span style="color:#00B050">Example: </span>  <span style="color:#00B050"> _6_ </span>  <span style="color:#00B050"> particles in </span>  <span style="color:#00B050"> _2_ </span>  <span style="color:#00B050"> branches</span>

Over a single time\-step\, some particles may cross into another branch

<span style="color:#FF0000"> __1\. Detect Crossings__ </span>

<span style="color:#FF0000">Find address of destination branch</span>

<span style="color:#00B050"> __2\. __ </span>  <span style="color:#00B050">Copy: </span>  <span style="color:#00B050"> __Branch → Buffer__ </span>

<span style="color:#00B050">\(Number of particles in branch\)</span>

<span style="color:#FF0000"> __Branch destination__ </span>

<span style="color:#0070C0"> __4\. __ </span>  <span style="color:#0070C0">Copy: </span>  <span style="color:#0070C0"> __Buffer → Branch__ </span>

<span style="color:#2F5597"> __Branch Particle Data__ </span>

WriteAddr =  <span style="color:#0070C0"> __atomicAdd__ </span> \(Np\,1\)

WriteAddr is Np before we added 1

_0_  __       __  _2_  __       __  _4_  __       __  _6_

_1_  __       __  _3_  __       __  _5_  __       __  _7_

<span style="color:#0070C0">atomicAdd</span>  <span style="color:#0070C0"> </span>  <span style="color:#0070C0"> __prohibits__ </span>  <span style="color:#0070C0"> other threads from </span>  <span style="color:#0070C0"> __read/write__ </span>  <span style="color:#0070C0"> until it returns\!</span>

_0           1        _  _ _  _  2           3_

_0           1        _  _ _  _  2           3_

