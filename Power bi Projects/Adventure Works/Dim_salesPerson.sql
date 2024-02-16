
create view view_Dim_salesPerson
as
select  
BusinessEntityID,
Title,
FirstName,
LastName,
JobTitle,
EmailAddress,
SalesLastYear

from [Sales].[vSalesPerson];






