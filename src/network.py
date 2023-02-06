import header
import logger
import torch.nn
import type

def configureRevision0(model, dataset, device):
    model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    return model

def configureRevision1(model, dataset, device):
    model.fc = torch.nn.Sequential(torch.nn.Dropout(p = header.train_dropout_probability), torch.nn.Linear(model.fc.in_features, len(dataset.classes)))
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    return model

def configure(revision, model, dataset, device):
    if revision == type.NetworkRevision.revision_0:
        return configureRevision0(model, dataset, device)
    elif revision == type.NetworkRevision.revision_1:
        return configureRevision1(model, dataset, device)
    else:
        logger.log_warn("Unknown network revision.")
        return model
