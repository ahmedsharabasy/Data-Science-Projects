
create view vieww_Fact_salesOrder
as
select 
	   SOD.[SalesOrderID]
      ,SOD.[SalesOrderDetailID]    
      ,SOD.[OrderQty]
      ,SOD.[ProductID]     
      ,SOD.[UnitPrice]      
	  ,CAST(FORMAT(SOH.[OrderDate],'yyyyMMdd') AS INT) [OrderDate]
	  ,CAST(FORMAT(SOH.[DueDate],'yyyyMMdd') AS INT) [DueDate]
	  ,CAST(FORMAT(SOH.[ShipDate],'yyyyMMdd') AS INT) [ShipDate]
	  ,SOH.[Status]
      ,SOH.[OnlineOrderFlag]
	  ,SOH.[CustomerID]
      ,SOH.[SalesPersonID]
      ,SOH.[TerritoryID]
      ,SOH.[ShipMethodID]
	  ,SOD.[LineTotal]
	  ,(SOD.LineTotal / SOH.SubTotal) * SOH.TaxAmt [TaxAmt]
	  ,(SOD.LineTotal / SOH.SubTotal) * SOH.[Freight] [Freight]
	  ,(SOD.LineTotal / SOH.SubTotal) * SOH.[TotalDue] [TotalDue]
  from [Sales].[SalesOrderDetail] SOD
  left join[Sales].[SalesOrderHeader] SOH
  on SOD.SalesOrderID = SOH.SalesOrderID