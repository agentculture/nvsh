# Platform detection sources

Device context nvsh attaches to a diagnosis — L4T/JetPack or DGX OS version,
CUDA/cuDNN/TensorRT/driver versions, unified memory, `nvpmodel`, container
runtime — is detected file-first and must report **what it found and how**,
never a guess. This page is the source-of-truth record issue #1 asks for:
one heading per target machine, each row naming the file or command read,
the value it yields, and the probe that verified it.

This is a **skeleton**. The value/source table for each machine is filled in
by a later task once `nvsh context`/`nvsh doctor` implement the detectors
described in the spec's "Platform detection" and "Jetson detection sources"
requirements; this page only fixes the headings and the shape of the table
so that work has a stable home.

## DGX Spark

| Value | Source | Notes |
|-------|--------|-------|
| _(to be filled in)_ | | |

## Jetson AGX Thor

| Value | Source | Notes |
|-------|--------|-------|
| _(to be filled in)_ | | |

## Jetson AGX Orin

| Value | Source | Notes |
|-------|--------|-------|
| _(to be filled in)_ | | |

## Reporting rule

Every value in every table above must be reported with the path or command
that produced it, or reported as **absent** when the source does not exist
on that machine (for example, Orin has no `/usr/local/cuda/version.json`
and no `tensorrt` Python module). nvsh never infers a value it did not read.
