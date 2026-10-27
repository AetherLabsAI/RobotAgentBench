FROM python@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84
COPY requirements.lock /opt/robotagentbench/requirements.lock
RUN python -m pip install --no-cache-dir -r /opt/robotagentbench/requirements.lock \
    && python -m pip check
LABEL org.robotagentbench.policy-sandbox="1" \
      org.robotagentbench.policy-lock="c74fdf44032127126f4467fad74db8e29eacb086415858243652e062fde5a822"
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
